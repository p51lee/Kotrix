package complex

import utils.R
import utils.StringVector
import utils.pseudoEquals
import kotlin.math.pow

/**
 * Represents complex, multidimensional tensor.
 *
 * @property shape
 * @property data
 * @property size
 * @property dim
 */
open class ComplexTensor(val shape: IntArray, val data: Array<ComplexDouble> =
    Array(shape.reduce {
            total, num ->
        if (num <= 0) throw IllegalArgumentException("ComplexTensor.init: Invalid shape")
        else total * num
    }
    ) { 0.0.R }
) {
    val size = data.size
    val dim = shape.size

    init {
        if (size != calculateSize(shape)) throw IllegalArgumentException("ComplexTensor.init: Invalid data length")
    }

    operator fun get(indices: IntArray): ComplexDouble {
        return if (indices.size != dim) throw IllegalArgumentException("ComplexTensor.get: Too many indices")
        else data[tensorIndicesToDataIndex(shape, indices)]
    }

    operator fun get(indexLong: Long): ComplexTensor {
        val index = indexLong.toInt()
        return when {
            index >= shape[0] -> throw IllegalArgumentException("ComplexTensor.get: Index out of bound")
            dim == 0 -> throw IllegalArgumentException("ComplexTensor.get: cannot get from 0-dimensional tensor. use [intArrayOf()] to get value.")
            else -> {
                val newShape = (1..shape.lastIndex).map { shape[it] }.toIntArray()
                val newTensorSize = newShape.reduce { total, num -> total * num }
                val dataIndexStart = index * newTensorSize
                val dataIndexEnd = (index + 1) * newTensorSize
                val newData = (dataIndexStart until dataIndexEnd).map { data[it] }.toTypedArray()
                ComplexTensor(newShape, newData)
            }
        }
    }

    operator fun set(indices: IntArray, value: ComplexDouble) {
        indices.forEachIndexed { index, it ->
            if (it < 0 || it >= shape[index]) throw IllegalArgumentException("ComplexTensor.set: Index out of bound")
        }
        data[tensorIndicesToDataIndex(shape, indices)] = value
    }

    open operator fun unaryPlus() = this

    open operator fun unaryMinus(): ComplexTensor {
        return ComplexTensor(shape, Array(size) {- data[it]})
    }

    operator fun plus(other: ComplexTensor): ComplexTensor {
        shape.forEachIndexed { index, it ->
            if (it != other.shape[index]) throw IllegalArgumentException("ComplexTensor.plus: Two tensors should have the same shape.")
        }
        val newData = Array(size) {
            data[it] + other.data[it]
        }
        return ComplexTensor(shape, newData)
    }

    operator fun minus(other: ComplexTensor): ComplexTensor {
        shape.forEachIndexed { index, it ->
            if (it != other.shape[index]) throw IllegalArgumentException("ComplexTensor.minus: Two tensors should have the same shape.")
        }
        val newData = Array(size) {
            data[it] - other.data[it]
        }
        return ComplexTensor(shape, newData)
    }

    operator fun times(other: ComplexTensor): ComplexTensor {
        return if (shape.last() != other.shape[0]) throw IllegalArgumentException("ComplexTensor.times: Invalid tensor product.")
        else {
            val newDim = dim - 1 + other.dim - 1
            val newShape = IntArray(newDim) {
                when {
                    it < dim - 1 -> shape[it]
                    else -> other.shape[it - (dim - 1) + 1]
                }
            }
            val newSize = newShape.reduce { tot, num -> tot * num }
            val newData = Array(newSize) { dataIndex ->
                val newIndices = dataIndexToTensorIndices(newShape, dataIndex)
                var sum = 0.0.R
                for (sumIndex in 0 until shape.last()) {
                    val indices1 = IntArray(dim) {
                        when {
                            it < dim - 1 -> newIndices[it]
                            else -> sumIndex
                        }
                    }
                    val indices2 = IntArray(other.dim) {
                        when (it) {
                            0 -> sumIndex
                            else -> newIndices[dim - 1 - 1 + it]
                        }
                    }
                    sum += this[indices1] * other[indices2]
                }
                sum
            }
            ComplexTensor(newShape, newData)
        }
    }

    open operator fun times(other: ComplexDouble): ComplexTensor {
        val newData = Array(size) {
            data[it] * other
        }
        return ComplexTensor(shape, newData)
    }

    open operator fun times(other: Number): ComplexTensor {
        val newData = Array(size) {
            data[it] * other.toDouble()
        }
        return ComplexTensor(shape, newData)
    }

    open operator fun div(other: Number): ComplexTensor {
        val newData = Array(size) {
            data[it] / other.toDouble()
        }
        return ComplexTensor(shape, newData)
    }

    open operator fun div(other: ComplexDouble): ComplexTensor {
        val newData = Array(size) {
            data[it] / other
        }
        return ComplexTensor(shape, newData)
    }

    override fun equals(other: Any?): Boolean {
        return if (other is ComplexTensor) {
            val shapeEq = this.shape.contentEquals(other.shape)
            val dataEq = this.data.contentEquals(other.data)
            shapeEq && dataEq
        } else false
    }

    fun pseudoEquals(other: ComplexTensor): Boolean {
        val shapeEq = this.shape.contentEquals(other.shape)
        val diffNorm = (this - other).frobeniusNormSquared()
        val dataPseudoEq = pseudoEquals((diffNorm / size).R, 0.R)
        return shapeEq && dataPseudoEq
    }

    fun toComplexMatrix(): ComplexMatrix {
        return when (dim) {
            1 -> {
                ComplexMatrix(1, shape[0], data)
            }
            2 -> {
                ComplexMatrix(shape[0], shape[1], data)
            }
            else -> throw IllegalStateException("ComplexTensor.toComplexMatrix: must be a 2 dimensional tensor, not $dim.")
        }
    }

    fun reshape(newShape: IntArray): ComplexTensor {
        var negOneIndex = -1
        var negOneCount = 0
        var acc = 1
        newShape.forEachIndexed { index, it ->
            when {
                it == -1 -> {
                    negOneCount++
                    negOneIndex = index
                }
                it > 0 -> {
                    acc *= it
                    if (size % acc != 0) throw IllegalArgumentException("ComplexTensor.reshape: invalid shape input")
                }
                else -> throw IllegalArgumentException("ComplexTensor.reshape: invalid shape input")
            }
        }
        if (negOneCount > 0) {
            newShape[negOneIndex] = size / acc
        }
        return ComplexTensor(newShape, data)
    }

    fun flatten(): ComplexTensor {
        return this.reshape(intArrayOf(-1))
    }

    fun concat(other: ComplexTensor, concatDim: Int): ComplexTensor {
        if (dim != other.dim || concatDim >= dim) throw IllegalArgumentException("ComplexTensor.concat: invalid dimension")
        else {
            shape.forEachIndexed { index, it ->
                if (index != concatDim && it != other.shape[index])
                    throw IllegalArgumentException("ComplexTensor.concat: two tensors must have same shape except for concat dimension")
            }
            val newShape = IntArray(shape.size) {
                when (it) {
                    concatDim -> shape[it] + other.shape[it]
                    else -> shape[it]
                }
            }
            val newSize = calculateSize(newShape)
            return ComplexTensor(newShape, Array(newSize) {
                val newIndices = dataIndexToTensorIndices(newShape, it)
                when {
                    newIndices[concatDim] < shape[concatDim] ->
                        this[newIndices]
                    else -> {
                        newIndices[concatDim] -= shape[concatDim]
                        other[newIndices]
                    }
                }
            })
        }
    }

    open fun map(lambda: (e: ComplexDouble) -> ComplexDouble): ComplexTensor {
        return ComplexTensor(shape, Array(size) {
            lambda(data[it])
        })
    }

    open fun copy() = ComplexTensor(shape, data.copyOf())

    fun frobeniusNormSquared(): Double {
        var frbNorm = 0.0
        data.forEach {
            frbNorm += it.abs().pow(2)
        }
        return frbNorm
    }

    private fun dataIndexToTensorIndices(newShape: IntArray, dataIndex: Int): IntArray {
        val retList = arrayListOf<Int>()
        newShape.foldRight(dataIndex) { it, acc ->
            retList.add(0, acc % it)
            acc / it
        }
        return retList.toIntArray()
    }

    private fun tensorIndicesToDataIndex(newShape: IntArray, tensorIndices: IntArray): Int {
        return tensorIndices.reduceIndexed { index, acc, tensorIndex ->
            if (tensorIndex >= newShape[index]) throw IllegalArgumentException("ComplexTensor.tensorIndicesToDataIndex: Index out of bound")
            (acc * newShape[index]) + tensorIndex
        }
    }

    private fun calculateSize(newShape: IntArray): Int {
        return newShape.reduce {
                total, num ->
            if (num <= 0) throw IllegalArgumentException("ComplexTensor.init: Invalid shape")
            else total * num
        }
    }

    private fun stackSuppl(other: ComplexTensor): ComplexTensor {
        return when (dim-other.dim) {
            0 -> {
                other.shape.forEachIndexed {index, it ->
                    if (this.shape[index] != it) throw IllegalArgumentException("ComplexTensor.stack: Cannot stack tensors with different shape")
                }
                ComplexTensor(intArrayOf(2) + other.shape, data + other.data)
            }
            1 -> {
                other.shape.forEachIndexed { index, it ->
                    if (this.shape[index + 1] != it) throw IllegalArgumentException("ComplexTensor.stack: Cannot stack tensors with different shape")
                }
                ComplexTensor(intArrayOf(shape[0]+1) + other.shape, data + other.data)
            }
            else -> throw IllegalArgumentException("ComplexTensor.stack: Cannot stack tensors with different shape")
        }
    }

    override fun toString(): String {
        return StringVector.build(this).toString() + "\n"
    }

    companion object {
        fun stack(tensors: Iterable<ComplexTensor>): ComplexTensor {
            val init = tensors.elementAt(0)
            return tensors.fold(init) { acc, tensor -> acc.stackSuppl(tensor) }
        }
    }
}