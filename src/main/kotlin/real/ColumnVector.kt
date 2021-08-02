package real

import complex.ComplexColumnVector
import utils.R
import kotlin.math.sqrt
import utils.times

class ColumnVector(val length: Int, data: DoubleArray = DoubleArray(length){0.0}) : Matrix(length, 1, data) {

    init {
        if (data.size != length)
            throw IllegalArgumentException("ColumnVector.init: length of the data is not valid")
    }

    constructor(length: Int, data: LongArray) : this(length, DoubleArray(length) { data[it].toDouble() })

    constructor(length: Int, data: FloatArray) : this(length, DoubleArray(length) { data[it].toDouble() })

    constructor(length: Int, data: IntArray) : this(length, DoubleArray(length) { data[it].toDouble() })

    constructor (length: Int, lambda: (i: Int) -> Number) : this(length, DoubleArray(length) { lambda(it).toDouble() })

    operator fun get(index: Int): Double {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("ColumnVector.get: Index out of bound")
        } else {
            return data[index]
        }
    }

    operator fun set(index: Int, value: Number) {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("ColumnVector.get: Index out of bound")
        } else {
            data[index] = value.toDouble()
        }
    }

    override operator fun unaryPlus() = this

    override operator fun unaryMinus(): ColumnVector {
        return ColumnVector(length, DoubleArray(length) {- data[it]})
    }

    operator fun plus(other: ColumnVector): ColumnVector {
        return if (length != other.length) {
            throw IllegalArgumentException("ColumnVector.plus: Two vectors should have the same size.")
        } else {
            val newData = DoubleArray(length) {
                data[it] + other.data[it]
            }
            ColumnVector(length, newData)
        }
    }

    operator fun minus(other: ColumnVector): ColumnVector {
        return if (length != other.length) {
            throw IllegalArgumentException("ColumnVector.minus: Two vectors should have the same size.")
        } else {
            val newData = DoubleArray(length) {
                data[it] - other.data[it]
            }
            ColumnVector(length, newData)
        }
    }

    override operator fun times(other: Number): ColumnVector {
        val newData = DoubleArray(length) {
            other.toDouble() * this[it]
        }
        return ColumnVector(length, newData)
    }

    override operator fun div(other: Number): ColumnVector {
        val newData = DoubleArray(length) {
            this[it] / other.toDouble()
        }
        return ColumnVector(length, newData)
    }

    override fun transpose(): RowVector {
        return RowVector(length, data)
    }

    fun getSubvector(indexStart: Int, indexEnd: Int): ColumnVector {
        return if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length) {
            throw IllegalArgumentException("ColumnVector.Subvector: Index out of bound")
        } else {
            val newSize = indexEnd - indexStart
            val newData = DoubleArray(newSize) {
                this[indexStart + it]
            }
            ColumnVector(newSize, newData)
        }
    }

    fun setSubvector(indexStart: Int, indexEnd: Int, other: ColumnVector) {
        val newSize = indexEnd - indexStart
        if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length || newSize != other.length) {
            throw IllegalArgumentException("ColumnVector.Subvector: Index out of bound")
        } else {
            other.data.forEachIndexed { index, element ->
                this[indexStart + index] = element
            }
        }
    }

    fun eltwiseMul(other: ColumnVector): ColumnVector {
        if (length != other.length)
            throw IllegalArgumentException("ColumnVector.eltwiseMul: Both operands must have the same size")
        return ColumnVector(length, DoubleArray(length) { this[it] * other[it] })
    }

    fun dotProduct(other: ColumnVector): Double {
        if (length != other.length)
            throw IllegalArgumentException("ColumnVector.dotProduct: Both operands must have the same size")
        var sum = 0.0
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun dotProduct(other: RowVector): Double {
        if (length != other.length)
            throw IllegalArgumentException("ColumnVector.dotProduct: Both operands must have the same size")
        var sum = 0.0
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun crossProduct(other: ColumnVector): ColumnVector {
        if (length != 3 || other.length != 3)
            throw IllegalArgumentException("ColumnVector.dotProduct: Both operands must be 3 dimensional vectors")
        else {
            return ColumnVector(length, doubleArrayOf(
                this[1] * other[2] - this[2] * other[1],
                this[2] * other[0] - this[0] * other[2],
                this[0] * other[1] - this[1] * other[0]
            ))
        }
    }

    fun replicate(length: Int): Matrix {
        if (length < 1) throw IllegalArgumentException("RowVector.replicate: length must be greater than 0.")
        return Matrix(this.length, length, DoubleArray(this.length * length) {
            val rowIndex = it / length
            this[rowIndex]
        })
    }

    override fun map(lambda: (e: Double) -> Number): ColumnVector {
        return ColumnVector(length, DoubleArray(length) {
            lambda(this[it]).toDouble()
        })
    }

    override fun toComplex(): ComplexColumnVector {
        return ComplexColumnVector(length, Array(length) {data[it].R})
    }

    override fun copy() = ColumnVector(length, data.copyOf())

    fun proj(direction: ColumnVector): ColumnVector {
        return (direction.dotProduct(this) / direction.frobeniusNormSquared()) * direction
    }

    fun normalize(): ColumnVector {
        return this / sqrt(this.frobeniusNormSquared())
    }
}
