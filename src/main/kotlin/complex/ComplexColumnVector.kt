package complex

import utils.R
import kotlin.math.sqrt

class ComplexColumnVector(val length: Int, data: Array<ComplexDouble> = Array(length){0.0.R}) : ComplexMatrix(length, 1, data) {
    init {
        if (data.size != length)
            throw IllegalArgumentException("ComplexColumnVector.init: length of the data is not valid")
    }
    constructor (length: Int, lambda: (i: Int) -> ComplexDouble) : this(length, Array(length) { lambda(it) })

    operator fun get(index: Int): ComplexDouble {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("ComplexColumnVector.get: Index out of bound")
        } else {
            return data[index]
        }
    }

    operator fun set(index: Int, value: ComplexDouble) {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("ColumnVector.get: Index out of bound")
        } else {
            data[index] = value
        }
    }

    override operator fun unaryPlus() = this

    override operator fun unaryMinus(): ComplexColumnVector {
        return ComplexColumnVector(length, Array(length) {- data[it]})
    }

    operator fun plus(other: ComplexColumnVector): ComplexColumnVector {
        return if (length != other.length) {
            throw IllegalArgumentException("ComplexColumnVector.plus: Two vectors should have the same size.")
        } else {
            val newData = Array(length) {
                data[it] + other.data[it]
            }
            ComplexColumnVector(length, newData)
        }
    }

    operator fun minus(other: ComplexColumnVector): ComplexColumnVector {
        return if (length != other.length) {
            throw IllegalArgumentException("ComplexColumnVector.minus: Two vectors should have the same size.")
        } else {
            val newData = Array(length) {
                data[it] - other.data[it]
            }
            ComplexColumnVector(length, newData)
        }
    }

    override operator fun times(other: ComplexDouble): ComplexColumnVector {
        val newData = Array(length) {
            this[it] * other
        }
        return ComplexColumnVector(length, newData)
    }

    override operator fun times(other: Number): ComplexColumnVector {
        val newData = Array(length) {
            this[it] * other
        }
        return ComplexColumnVector(length, newData)
    }

    override operator fun div(other: ComplexDouble): ComplexColumnVector {
        val newData = Array(length) {
            this[it] / other
        }
        return ComplexColumnVector(length, newData)
    }

    override operator fun div(other: Number): ComplexColumnVector {
        val newData = Array(length) {
            this[it] / other.toDouble()
        }
        return ComplexColumnVector(length, newData)
    }

    override fun transpose(): ComplexRowVector {
        return ComplexRowVector(length, data)
    }

    fun getSubvector(indexStart: Int, indexEnd: Int): ComplexColumnVector {
        return if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length) {
            throw IllegalArgumentException("ComplexColumnVector.Subvector: Index out of bound")
        } else {
            val newSize = indexEnd - indexStart
            val newData = Array(newSize) {
                this[indexStart + it]
            }
            ComplexColumnVector(newSize, newData)
        }
    }

    fun setSubvector(indexStart: Int, indexEnd: Int, other: ComplexColumnVector) {
        val newSize = indexEnd - indexStart
        if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length || newSize != other.length) {
            throw IllegalArgumentException("ComplexColumnVector.Subvector: Index out of bound")
        } else {
            other.data.forEachIndexed { index, element ->
                this[indexStart + index] = element
            }
        }
    }

    fun eltwiseMul(other: ComplexColumnVector): ComplexColumnVector {
        if (length != other.length)
            throw IllegalArgumentException("ComplexColumnVector.eltwiseMul: Both operands must have the same size")
        return ComplexColumnVector(length, Array(length) { this[it] * other[it] })
    }

    fun dotProduct(other: ComplexColumnVector): ComplexDouble {
        if (length != other.length)
            throw IllegalArgumentException("ComplexColumnVector.dotProduct: Both operands must have the same size")
        var sum = 0.0.R
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun dotProduct(other: ComplexRowVector): ComplexDouble {
        if (length != other.length)
            throw IllegalArgumentException("ComplexColumnVector.dotProduct: Both operands must have the same size")
        var sum = 0.0.R
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun crossProduct(other: ComplexColumnVector): ComplexColumnVector {
        if (length != 3 || other.length != 3)
            throw IllegalArgumentException("ComplexColumnVector.dotProduct: Both operands must be 3 dimensional vectors")
        else {
            return ComplexColumnVector(length, arrayOf(
                this[1] * other[2] - this[2] * other[1],
                this[2] * other[0] - this[0] * other[2],
                this[0] * other[1] - this[1] * other[0]
            ))
        }
    }

    fun replicate(length: Int): ComplexMatrix {
        if (length < 1) throw IllegalArgumentException("ComplexRowVector.replicate: length must be greater than 0.")
        return ComplexMatrix(this.length, length, Array(this.length * length) {
            val rowIndex = it / length
            this[rowIndex]
        })
    }

    fun normalize(): ComplexColumnVector {
        return this / sqrt(this.frobeniusNormSquared())
    }

    fun proj(direction: ComplexColumnVector): ComplexColumnVector {
        return (direction.dotProduct(this) / direction.frobeniusNormSquared()) * direction
    }

    override fun map(lambda: (e: ComplexDouble) -> ComplexDouble): ComplexColumnVector {
        return ComplexColumnVector(length, Array(length) {
            lambda(this[it])
        })
    }

    override fun copy() = ComplexColumnVector(length, data.copyOf())
}
