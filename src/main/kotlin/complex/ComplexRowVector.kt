package complex

import real.ColumnVector
import real.Matrix
import real.RowVector
import real.Tensor
import utils.R

/**
 * Represents a complex row vector. It is also a [Matrix] with shape 1 * [length].
 *
 * @property length length of a vector.
 * @constructor Creates a new row vector. If [data] is not given, it will generate a zero vector.
 *
 * @param data must fit into the length of this vector.
 */
class ComplexRowVector(val length: Int, data: Array<ComplexDouble> = Array(length){0.0.R}): ComplexMatrix(1, length, data) {
    init {
        if (data.size != length)
            throw IllegalArgumentException("ComplexRowVector.init: length of the data is not valid")
    }

    /**
     * It is possible to set a value using lambda function.
     */
    constructor(length: Int, lambda: (i: Int) -> ComplexDouble) : this(length, Array(length) { lambda(it) })

    /**
     * If [data] is given, [length] is no longer necessary.
     */
    constructor(data: Array<ComplexDouble>) : this(data.size, data)

    operator fun get(index: Int): ComplexDouble {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("ComplexRowVector.get: Index out of bound")
        } else {
            return data[index]
        }
    }

    operator fun set(index: Int, value: ComplexDouble) {
        if (index < 0 || index >= length) {
            throw IllegalArgumentException("ComplexRowVector.get: Index out of bound")
        } else {
            data[index] = value
        }
    }

    override operator fun unaryPlus() = this

    override operator fun unaryMinus(): ComplexRowVector {
        return ComplexRowVector(length, Array(length) {- data[it]})
    }

    operator fun plus(other: RowVector): ComplexRowVector {
        return if (length != other.length) {
            throw IllegalArgumentException("ComplexRowVector.plus: Two vectors should have the same size.")
        } else {
            val newData = Array(length) {
                data[it] + other.data[it]
            }
            ComplexRowVector(length, newData)
        }
    }

    operator fun minus(other: RowVector): ComplexRowVector {
        return if (length != other.length) {
            throw IllegalArgumentException("RowVector.minus: Two vectors should have the same size.")
        } else {
            val newData = Array(length) {
                data[it] - other.data[it]
            }
            ComplexRowVector(length, newData)
        }
    }

    override operator fun times(other: ComplexMatrix): ComplexRowVector {
        return if (length != other.rows) {
            throw IllegalArgumentException("ComplexRowVector.times: Illegal Matrix multiplication.")
        } else {
            val newData = Array(other.cols) {
                var sum = 0.0.R
                for (i in 0 until length) {
                    sum += this[i] * other[i, it]
                }
                sum
            }
            ComplexRowVector(other.cols, newData)
        }
    }

    override operator fun times(other: Number): ComplexRowVector {
        val newData = Array(length) {
            this[it] * other
        }
        return ComplexRowVector(length, newData)
    }

    override operator fun times(other: ComplexDouble): ComplexRowVector {
        val newData = Array(length) {
            this[it] * other
        }
        return ComplexRowVector(length, newData)
    }

    override operator fun div(other: Number): ComplexRowVector {
        val newData = Array(length) {
            this[it] / other.toDouble()
        }
        return ComplexRowVector(length, newData)
    }

    override operator fun div(other: ComplexDouble): ComplexRowVector {
        val newData = Array(length) {
            this[it] / other
        }
        return ComplexRowVector(length, newData)
    }

    /**
     * Same as [ComplexMatrix.transpose] but returns [ComplexColumnVector].
     *
     * @return transposed vector.
     */
    override fun transpose(): ComplexColumnVector {
        return ComplexColumnVector(length, data)
    }

    /**
     * Get a subvector by slicing this vector.
     *
     * @param indexStart where slice starts.
     * @param indexEnd where slice ends. (not included)
     * @return sliced subvector.
     */
    fun getSubvector(indexStart: Int, indexEnd: Int): ComplexRowVector {
        return if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length) {
            throw IllegalArgumentException("ComplexRowVector.Subvector: Index out of bound")
        } else {
            val newSize = indexEnd - indexStart
            val newData =Array(newSize) {
                this[indexStart + it]
            }
            ComplexRowVector(newSize, newData)
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
    fun setSubvector(indexStart: Int, indexEnd: Int, other: ComplexRowVector) {
        val newSize = indexEnd - indexStart
        if (indexStart < 0 || indexStart >= indexEnd || indexEnd > length || newSize != other.length) {
            throw IllegalArgumentException("ComplexRowVector.Subvector: Index out of bound")
        } else {
            other.data.forEachIndexed { index, element ->
                this[indexStart + index] = element
            }
        }
    }

    /**
     * Same as [ComplexMatrix.eltwiseMul] but returns [ComplexRowVector].
     *
     * @param other must have the same length as this vector.
     * @return multiplied vector.
     */
    fun eltwiseMul(other: ComplexRowVector): ComplexRowVector {
        if (length != other.length)
            throw IllegalArgumentException("ComplexRowVector.eltwiseMul: Both operands must have the same size")
        return ComplexRowVector(length, Array(length) { this[it] * other[it] })
    }

    /**
     * Dot product.
     *
     * @param other must have the same length as this vector.
     * @return result of the product.
     */
    fun dotProduct(other: ComplexRowVector): ComplexDouble {
        if (length != other.length)
            throw IllegalArgumentException("ComplexRowVector.dotProduct: Both operands must have the same size")
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
    fun dotProduct(other: ColumnVector): ComplexDouble {
        if (length != other.length)
            throw IllegalArgumentException("ComplexRowVector.dotProduct: Both operands must have the same size")
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
    fun crossProduct(other: ComplexRowVector): ComplexRowVector {
        if (length != 3 || other.length != 3)
            throw IllegalArgumentException("ComplexRowVector.dotProduct: Both operands must be 3 dimensional vectors")
        else {
            return ComplexRowVector(length, arrayOf(
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
        return ComplexMatrix(n, this.length, Array(n * this.length) {
            val colIndex = it % this.length
            this[colIndex]
        })
    }

    /**
     * Same as [Tensor.map] but returns [ComplexRowVector].
     *
     * @param lambda mapping function.
     * @return a newly mapped vector.
     */
    override fun map(lambda: (e: ComplexDouble) -> ComplexDouble): ComplexRowVector {
        return ComplexRowVector(length, Array(length) {
            lambda(this[it])
        })
    }

    /**
     * Same as [ComplexTensor.copy] but returns [ComplexRowVector]
     *
     * @return a new [ComplexRowVector] instance with the same shape and data.
     */
    override fun copy() = ComplexRowVector(length, data.copyOf())
}
