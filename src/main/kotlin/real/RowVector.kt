package real

import complex.ComplexRowVector
import utils.R
import kotlin.math.sqrt
import utils.times

class RowVector(val length: Int, data: DoubleArray = DoubleArray(length){0.0}): Matrix(1, length, data) {
    init {
        if (data.size != length)
            throw IllegalArgumentException("RowVector.init: length of the data is not valid")
    }

    constructor(length: Int, data: LongArray) : this(length, DoubleArray(length) { data[it].toDouble() })

    constructor(length: Int, data: FloatArray) : this(length, DoubleArray(length) { data[it].toDouble() })

    constructor(length: Int, data: IntArray) : this(length, DoubleArray(length) { data[it].toDouble() })

    constructor (length: Int, lambda: (i: Int) -> Number) : this(length, DoubleArray(length) { lambda(it).toDouble() })

    operator fun get(index: Int): Double {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("RowVector.get: Index out of bound")
        } else {
            return data[index]
        }
    }

    operator fun set(index: Int, value: Number) {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("RowVector.get: Index out of bound")
        } else {
            data[index] = value.toDouble()
        }
    }

    override operator fun unaryPlus() = this

    override operator fun unaryMinus(): RowVector {
        return RowVector(length, DoubleArray(length) {- data[it]})
    }

    operator fun plus(other: RowVector): RowVector {
        return if (length != other.length) {
            throw IllegalArgumentException("RowVector.plus: Two vectors should have the same size.")
        } else {
            val newData = DoubleArray(length) {
                data[it] + other.data[it]
            }
            RowVector(length, newData)
        }
    }

    operator fun minus(other: RowVector): RowVector {
        return if (length != other.length) {
            throw IllegalArgumentException("RowVector.minus: Two vectors should have the same size.")
        } else {
            val newData = DoubleArray(length) {
                data[it] - other.data[it]
            }
            RowVector(length, newData)
        }
    }

    override operator fun times(other: Matrix): RowVector {
        return if (length != other.rows) {
            throw IllegalArgumentException("RowVector.times: Illegal Matrix multiplication.")
        } else {
            val newData = DoubleArray(other.cols) {
                var sum = 0.0
                for (i in 0 until length) {
                    sum += this[i] * other[i, it]
                }
                sum
            }
            RowVector(other.cols, newData)
        }
    }

    override operator fun times(other: Number): RowVector {
        val newData = DoubleArray(length) {
            other.toDouble() * this[it]
        }
        return RowVector(length, newData)
    }

    override operator fun div(other: Number): RowVector {
        val newData = DoubleArray(length) {
            this[it] / other.toDouble()
        }
        return RowVector(length, newData)
    }

    override fun transpose(): ColumnVector {
        return ColumnVector(length, data)
    }

    fun getSubvector(indexStart: Int, indexEnd: Int): RowVector {
        return if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length) {
            throw IllegalArgumentException("RowVector.Subvector: Index out of bound")
        } else {
            val newSize = indexEnd - indexStart
            val newData = DoubleArray(newSize) {
                this[indexStart + it]
            }
            RowVector(newSize, newData)
        }
    }

    fun setSubvector(indexStart: Int, indexEnd: Int, other: RowVector) {
        val newSize = indexEnd - indexStart
        if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length || newSize != other.length) {
            throw IllegalArgumentException("RowVector.Subvector: Index out of bound")
        } else {
            other.data.forEachIndexed { index, element ->
                this[indexStart + index] = element
            }
        }
    }

    fun eltwiseMul(other: RowVector): RowVector {
        if (length != other.length)
            throw IllegalArgumentException("RowVector.eltwiseMul: Both operands must have the same size")
        return RowVector(length, DoubleArray(length) { this[it] * other[it] })
    }

    fun dotProduct(other: RowVector): Double {
        if (length != other.length)
            throw IllegalArgumentException("RowVector.dotProduct: Both operands must have the same size")
        var sum = 0.0
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun dotProduct(other: ColumnVector): Double {
        if (length != other.length)
            throw IllegalArgumentException("RowVector.dotProduct: Both operands must have the same size")
        var sum = 0.0
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    fun crossProduct(other: RowVector): RowVector {
        if (length != 3 || other.length != 3)
            throw IllegalArgumentException("RowVector.dotProduct: Both operands must be 3 dimensional vectors")
        else {
            return RowVector(length, doubleArrayOf(
                this[1] * other[2] - this[2] * other[1],
                this[2] * other[0] - this[0] * other[2],
                this[0] * other[1] - this[1] * other[0]
            ))
        }
    }

    fun replicate(length: Int): Matrix {
        if (length < 1) throw IllegalArgumentException("RowVector.replicate: length must be greater than 0.")
        return Matrix(length, this.length, DoubleArray(length * this.length) {
            val colIndex = it % this.length
            this[colIndex]
        })
    }

    override fun map(lambda: (e: Double) -> Number): RowVector {
        return RowVector(length, DoubleArray(length) {
            lambda(this[it]).toDouble()
        })
    }

    override fun toComplex(): ComplexRowVector {
        return ComplexRowVector(length, Array(length) {data[it].R})
    }

    override fun copy() = RowVector(length, data.copyOf())

    fun proj(direction: RowVector): RowVector {
        return (direction.dotProduct(this)) / direction.frobeniusNormSquared() * direction
    }

    fun normalize(): RowVector {
        return this / sqrt(this.frobeniusNormSquared())
    }
}
