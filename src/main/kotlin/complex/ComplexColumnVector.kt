package complex

import real.Matrix
import real.Tensor
import utils.R
import kotlin.math.sqrt

/**
 * Represents a complex column vector. It is also a [Matrix] with shape [length] * 1.
 *
 * @property length length of a vector.
 * @constructor Creates a new column vector. If [data] is not given, it will generate a zero vector.
 *
 * @param data must fit into the length of this vector.
 */
class ComplexColumnVector(val length: Int, data: Array<ComplexDouble> = Array(length){0.0.R}) : ComplexMatrix(length, 1, data) {
    init {
        if (data.size != length)
            throw IllegalArgumentException("ComplexColumnVector.init: length of the data is not valid")
    }

    /**
     * It is possible to set a value using lambda function.
     */
    constructor (length: Int, lambda: (i: Int) -> ComplexDouble) : this(length, Array(length) { lambda(it) })

    /**
     * If [data] is given, [length] is no longer necessary.
     */
    constructor(data: Array<ComplexDouble>) : this(data.size, data)

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

    /**
     * Same as [ComplexMatrix.transpose] but returns [ComplexRowVector].
     *
     * @return transposed vector.
     */
    override fun transpose(): ComplexRowVector {
        return ComplexRowVector(length, data)
    }

    /**
     * Get a subvector by slicing this vector.
     *
     * @param indexStart where slice starts.
     * @param indexEnd where slice ends. (not included)
     * @return sliced subvector.
     */
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

    /**
     * Set a subvector by substituting [other] to desired position.
     * The length of [other] must match the position you described.
     *
     * @param indexStart where substitution starts.
     * @param indexEnd where substitution ends. (not included)
     * @param other new subvector.
     */
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

    /**
     * Same as [ComplexMatrix.eltwiseMul] but returns [ComplexColumnVector].
     *
     * @param other must have the same length as this vector.
     * @return multiplied vector.
     */
    fun eltwiseMul(other: ComplexColumnVector): ComplexColumnVector {
        if (length != other.length)
            throw IllegalArgumentException("ComplexColumnVector.eltwiseMul: Both operands must have the same size")
        return ComplexColumnVector(length, Array(length) { this[it] * other[it] })
    }

    /**
     * Dot product.
     *
     * @param other must have the same length as this vector.
     * @return result of the product.
     */
    fun dotProduct(other: ComplexColumnVector): ComplexDouble {
        if (length != other.length)
            throw IllegalArgumentException("ComplexColumnVector.dotProduct: Both operands must have the same size")
        var sum = 0.0.R
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    /**
     * Dot product.
     *
     * @param other must have the same length as this vector.
     * @return result of the product.
     */
    fun dotProduct(other: ComplexRowVector): ComplexDouble {
        if (length != other.length)
            throw IllegalArgumentException("ComplexColumnVector.dotProduct: Both operands must have the same size")
        var sum = 0.0.R
        for (i in 0 until length) {
            sum += this[i] * other[i]
        }
        return sum
    }

    /**
     * Cross product. Available for 3-dimensional vectors only.
     *
     * @param other 3-dimensional vector.
     * @return result of the product.
     */
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

    /**
     * Concatenate the copies of this vector [n] times.
     *
     * @param n
     * @return result of the concatenation.
     */
    fun replicate(n: Int): ComplexMatrix {
        if (n < 1) throw IllegalArgumentException("ComplexRowVector.replicate: length must be greater than 0.")
        return ComplexMatrix(this.length, n, Array(this.length * n) {
            val rowIndex = it / n
            this[rowIndex]
        })
    }

    /**
     * Same as [Tensor.map] but returns [ComplexColumnVector].
     *
     * @param lambda mapping function.
     * @return a newly mapped vector.
     */
    override fun map(lambda: (e: ComplexDouble) -> ComplexDouble): ComplexColumnVector {
        return ComplexColumnVector(length, Array(length) {
            lambda(this[it])
        })
    }

    /**
     * Same as [ComplexTensor.copy] but returns [ComplexColumnVector]
     *
     * @return a new [ComplexColumnVector] instance with the same shape and data.
     */
    override fun copy() = ComplexColumnVector(length, data.copyOf())

    /**
     * Normalize this vector.
     *
     * @return normalized vector.
     */
    fun normalize(): ComplexColumnVector {
        return this / sqrt(this.frobeniusNormSquared())
    }

    /**
     * Project this vector onto [direction].
     *
     * @param direction direction vector.
     * @return result of the vector projection.
     */
    fun proj(direction: ComplexColumnVector): ComplexColumnVector {
        return (direction.dotProduct(this) / direction.frobeniusNormSquared()) * direction
    }
}
