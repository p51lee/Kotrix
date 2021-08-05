package complex

import operations.determinant
import real.ColumnVector
import real.Matrix
import real.Tensor
import utils.R
import utils.sum
import kotlin.math.pow
import utils.times

/**
 * Represents complex matrix. [ComplexMatrix] class is a subclass of [ComplexTensor] class.
 *
 * @property rows number of rows.
 * @property cols number of columns.
 * @constructor Creates a new matrix. If [data] is not given, it will generate a zero matrix.
 *
 * @param data must fit into the shape of this matrix.
 */
open class ComplexMatrix (val rows: Int, val cols: Int, data: Array<ComplexDouble> = Array(rows * cols) { 0.0.R }) :
    ComplexTensor(intArrayOf(rows, cols), data) {

    /**
     * It is possible to set a value using lambda function.
     */
    constructor(rows2: Int, cols2: Int, lambda: (i: Int, j: Int) -> ComplexDouble) : this(rows2, cols2,
        Array(rows2 * cols2) {
            val rowIndex = it / cols2
            val colIndex = it % cols2
            lambda(rowIndex, colIndex)
        })

    operator fun get(rowIndex: Int, colIndex: Int) : ComplexDouble {
        if (rowIndex < 0 || colIndex < 0 || rowIndex >= rows || colIndex >= cols) {
            throw IllegalArgumentException("ComplexMatrix.get: Index out of bound")
        } else {
            return data[rowIndex * cols + colIndex]
        }
    }

    operator fun set(rowIndex: Int, colIndex: Int, value: ComplexDouble) {
        if (rowIndex < 0 || colIndex < 0 || rowIndex >= rows || colIndex >= cols) {
            throw IllegalArgumentException("Matrix.set: Index out of bound")
        } else {
            data[rowIndex * cols + colIndex] = value
        }
    }

    override operator fun unaryPlus() = this

    override operator fun unaryMinus(): ComplexMatrix {
        return ComplexMatrix(rows, cols, Array(rows * cols) {- data[it]})
    }

    operator fun plus(other: ComplexMatrix): ComplexMatrix {
        return if (rows != other.rows || cols != other.cols) {
            throw IllegalArgumentException("ComplexMatrix.plus: Two matrices should have the same shape.")
        } else {
            val newData = Array(rows * cols) {
                data[it] + other.data[it]
            }
            ComplexMatrix(rows, cols, newData)
        }
    }

    operator fun minus(other: ComplexMatrix): ComplexMatrix {
        return if (rows != other.rows || cols != other.cols) {
            throw IllegalArgumentException("ComplexMatrix.minus: Two matrices should have the same shape.")
        } else {
            val newData = Array(rows * cols) {
                data[it] - other.data[it]
            }
            ComplexMatrix(rows, cols, newData)
        }
    }

    open operator fun times(other: ComplexMatrix): ComplexMatrix {
        return if (cols != other.rows) {
            throw IllegalArgumentException("ComplexMatrix.times: Illegal Matrix multiplication.")
        } else {
            val newData = Array(rows * other.cols) {
                val rowIndex = it / other.cols
                val colIndex = it % other.cols
                var sum = 0.0.R
                for (i in 0 until this.cols) {
                    sum += this[rowIndex, i] * other[i, colIndex]
                }
                sum
            }
            ComplexMatrix(rows, other.cols, newData)
        }
    }

    operator fun times(other: ComplexColumnVector): ComplexColumnVector {
        return if (cols != other.length) {
            throw IllegalArgumentException("ComplexMatrix.times: Illegal Matrix multiplication.")
        } else {
            val newData = Array(rows * 1) {
                var sum = 0.0.R
                for (i in 0 until this.cols) {
                    sum += this[it, i] * other[i]
                }
                sum
            }
            ComplexColumnVector(rows, newData)
        }
    }

    operator fun times(other: ColumnVector): ComplexColumnVector {
        return if (cols != other.length) {
            throw IllegalArgumentException("ComplexMatrix.times: Illegal Matrix multiplication.")
        } else {
            val newData = Array(rows * 1) {
                var sum = 0.0.R
                for (i in 0 until this.cols) {
                    sum += this[it, i] * other[i]
                }
                sum
            }
            ComplexColumnVector(rows, newData)
        }
    }

    override operator fun times(other: ComplexDouble): ComplexMatrix {
        val newData = Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            other * this[rowIndex, colIndex]
        }
        return ComplexMatrix(rows, cols, newData)
    }

    override operator fun times(other: Number): ComplexMatrix {
        val newData = Array(rows * cols) {
            data[it] * other.toDouble()
        }
        return ComplexMatrix(rows, cols, newData)
    }

    override operator fun div(other: ComplexDouble): ComplexMatrix {
        val newData = Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            this[rowIndex, colIndex] / other
        }
        return ComplexMatrix(rows, cols, newData)
    }

    /**
     * Transpose this Matrix.
     *
     * @return transposed matrix.
     */
    open fun transpose(): ComplexMatrix {
        val newData = Array(rows * cols) {
            val transposedRowIndex = it / rows
            val transposedColIndex = it % rows
            this[transposedColIndex, transposedRowIndex]
        }
        return ComplexMatrix(cols, rows, newData)
    }

    /**
     * Calculate conjugate transpose of this matrix.
     *
     * @return conjugate transpose of this matrix.
     */
    open fun conjTrans(): ComplexMatrix {
        val newData = Array(rows * cols) {
            val transposedRowIndex = it / rows
            val transposedColIndex = it % rows
            this[transposedColIndex, transposedRowIndex].conj()
        }
        return ComplexMatrix(cols, rows, newData)
    }

    /**
     * Calculate adjoint matrix of this matrix.
     *
     * @return adjoint matrix.
     */
    fun adjoint() : ComplexMatrix {
        if (rows != cols) throw IllegalArgumentException("ComplexMatrix.adjointMatrix: Only available for square matrices")
        val newData = Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            val sign = (-1.0).pow(rowIndex + colIndex)
            val cofactorDet = this.minorMatrix(rowIndex, colIndex).determinant()
            sign * cofactorDet
        }
        return ComplexMatrix(rows, cols, newData).transpose()
    }

    /**
     * Get a submatrix by slicing this matrix.
     *
     * @param rowIndexStart row index starting point.
     * @param rowIndexEnd row index ending point. (not included)
     * @param colIndexStart column index starting point.
     * @param colIndexEnd column index ending point. (not included)
     * @return sliced submatrix.
     */
    open fun getSubmatrix(rowIndexStart: Int, rowIndexEnd: Int, colIndexStart: Int, colIndexEnd: Int): ComplexMatrix {
        return if (rowIndexStart < 0 || colIndexStart < 0 || rowIndexStart >= rowIndexEnd || colIndexStart >= colIndexEnd
            || rowIndexEnd > rows || colIndexEnd > cols) {
            throw IllegalArgumentException("ComplexMatrix.Submatrix: Index out of bound")
        } else {
            val newRows = rowIndexEnd - rowIndexStart
            val newCols = colIndexEnd - colIndexStart
            val newData = Array(newRows * newCols) {
                val newRowIndex = it / newCols
                val newColIndex = it % newCols
                this[rowIndexStart + newRowIndex, colIndexStart + newColIndex]
            }
            ComplexMatrix(newRows, newCols, newData)
        }
    }

    /**
     * Set a submatrix by substituting [other] to desired position.
     * The shape of [other] must match the position you described.
     *
     * @param rowIndexStart row index starting point.
     * @param rowIndexEnd row index ending point. (not included)
     * @param colIndexStart column index starting point.
     * @param colIndexEnd column index ending point. (not included)
     * @param other new submatrix.
     */
    open fun setSubmatrix(rowIndexStart: Int, rowIndexEnd: Int, colIndexStart: Int, colIndexEnd: Int, other: ComplexMatrix) {
        val newRows = rowIndexEnd - rowIndexStart
        val newCols = colIndexEnd - colIndexStart
        if (rowIndexStart < 0 || colIndexStart < 0 || rowIndexStart >= rowIndexEnd || colIndexStart >= colIndexEnd
            || rowIndexEnd > rows || colIndexEnd > cols || newRows != other.rows || newCols != other.cols) {
            throw IllegalArgumentException("ComplexMatrix.Submatrix: Index out of bound")
        } else {
            other.data.forEachIndexed { index, element ->
                val otherRowIndex = index / other.cols
                val otherColIndex = index % other.cols
                this[rowIndexStart + otherRowIndex, colIndexStart + otherColIndex] = element
            }
        }
    }

    /**
     * Get a minor matrix.
     * Minors are obtained by removing just one row and one column from square matrices. (first minors)
     *
     * @param rowIndex
     * @param colIndex
     * @return
     */
    fun minorMatrix(rowIndex: Int, colIndex: Int) : ComplexMatrix {
        return if (rows < 2 || cols < 2 || rowIndex >= rows || colIndex >= cols) {
            throw IllegalArgumentException("ComplexMatrix.cofactorMatrix: Index out of bound")
        } else {
            val newData = Array((rows - 1) * (cols - 1)) {
                var cofactorRowIndex = it / (cols - 1)
                var cofactorColIndex = it % (cols - 1)
                if (cofactorRowIndex >= rowIndex) cofactorRowIndex += 1
                if (cofactorColIndex >= colIndex) cofactorColIndex += 1
                this[cofactorRowIndex, cofactorColIndex]
            }
            ComplexMatrix(rows - 1, cols - 1, newData)
        }
    }

    /**
     * Do a row switching transformation.
     *
     * @param rowIndex1
     * @param rowIndex2
     * @return new matrix that [rowIndex1]th row and [rowIndex2]th row has exchanged.
     */
    fun switchRow(rowIndex1: Int, rowIndex2: Int): ComplexMatrix {
        return if (rowIndex1 < 0 || rowIndex2 < 0 || rowIndex1 >= rows || rowIndex2 >= rows) {
            throw IllegalArgumentException("ComplexMatrix.switchRow: Index out of bound")
        } else if (rowIndex1 == rowIndex2) {
            this
        } else {
            val newData = Array(rows * cols) {
                val newRowIndex = it / cols
                val newColIndex = it % cols
                when (newRowIndex) {
                    rowIndex1 -> this[rowIndex2, newColIndex]
                    rowIndex2 -> this[rowIndex1, newColIndex]
                    else -> this[newRowIndex, newColIndex]
                }
            }
            ComplexMatrix(rows, cols, newData)
        }
    }

    /**
     * Do a row addition transformation; add dstRow multiplied by [fraction] to srcRow.
     *
     * @param srcRowIndex
     * @param dstRowIndex
     * @param fraction
     * @return new transformed matrix.
     */
    fun addRow(srcRowIndex: Int, dstRowIndex: Int, fraction: Number): ComplexMatrix {
        return if (srcRowIndex < 0 || dstRowIndex < 0 || srcRowIndex >= rows || dstRowIndex >= rows) {
            throw IllegalArgumentException("ComplexMatrix.addRow: Index out of bound")
        } else if (srcRowIndex == dstRowIndex) {
            throw IllegalArgumentException("ComplexMatrix.addRow: srcRow and dstRow must be different")
        } else {
            val newData = Array(rows * cols) {
                val newRowIndex = it / cols
                val newColIndex = it % cols
                when (newRowIndex) {
                    dstRowIndex -> this[dstRowIndex, newColIndex] + fraction.toDouble() * this[srcRowIndex, newColIndex]
                    else -> this[newRowIndex, newColIndex]
                }
            }
            ComplexMatrix(rows, cols, newData)
        }
    }

    /**
     * Concatenate to other matrix.
     *
     * @param other matrix to be concatenated.
     * @param concatDim dimension to which other matrix concatenate.
     * @return concatenated matrix.
     */
    fun concat(other: ComplexMatrix, concatDim: Int) : ComplexMatrix {
        return when (concatDim) {
            0 -> {
                if (cols != other.cols) throw IllegalArgumentException("ComplexMatrix.concat: number of columns does not match")
                val newRows = rows + other.rows
                val newCols = cols
                val newData = Array(newRows * newCols) {
                    val newRowIndex = it / newCols
                    val newColIndex = it % newCols
                    if (newRowIndex < rows) {
                        this[newRowIndex, newColIndex]
                    } else {
                        other[newRowIndex - rows, newColIndex]
                    }
                }
                ComplexMatrix(newRows, newCols, newData)
            }
            1 -> {
                if (rows != other.rows) throw IllegalArgumentException("ComplexMatrix.concat: number of rows does not match")
                val newRows = rows
                val newCols = cols + other.cols
                val newData = Array(newRows * newCols) {
                    val newRowIndex = it / newCols
                    val newColIndex = it % newCols
                    if (newColIndex < cols) {
                        this[newRowIndex, newColIndex]
                    } else {
                        other[newRowIndex, newColIndex - cols]
                    }
                }
                ComplexMatrix(newRows, newCols, newData)
            }
            else -> throw IllegalArgumentException("ComplexMatrix.concat: dim must be 0 or 1")
        }
    }

    /**
     * Calculate the squared norm of each column vector.
     *
     * @return row vector of each squared norm.
     */
    fun colVecNormSq(): ComplexRowVector {
        val newData = Array(1 * cols) {
            var norm = 0.0.R
            for (rowIndex in 0 until rows) {
                norm += this[rowIndex, it].abs().pow(2)
            }
            norm
        }
        return ComplexRowVector(cols, newData)
    }

    /**
     * Calculate the squared norm of each row vector.
     *
     * @return column vector of each squared norm.
     */
    fun rowVecNormSq(): ComplexColumnVector {
        val newData = Array(rows * 1) {
            var norm = 0.0.R
            for (colIndex in 0 until cols) {
                norm += this[it, colIndex].abs().pow(2)
            }
            norm
        }
        return ComplexColumnVector(rows, newData)
    }

    /**
     * Calculate the sum of all values in this matrix.
     *
     * @return the sum.
     */
    fun sum(): ComplexDouble {
        var sum = 0.0.R
        data.forEach { sum += it }
        return sum
    }

    /**
     * Element-wise multiplication. [other] must have the same shape as this matrix.
     *
     * @param other
     * @return multiplied matrix.
     */
    fun eltwiseMul(other: ComplexMatrix): ComplexMatrix {
        if (rows != other.rows || cols != other.cols)
            throw IllegalArgumentException("ComplexMatrix.eltwiseMul: Both operands must have the same shape")
        return ComplexMatrix(rows, cols, Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            this[rowIndex, colIndex] * other[rowIndex, colIndex]
        })
    }

    /**
     * Calculate the mean value of each row.
     *
     * @return column vector of each mean.
     */
    fun rowWiseMean(): ComplexColumnVector {
        return ComplexColumnVector(rows, Array(rows) {
            var rowSum = 0.0.R
            for (colIndex in 0 until cols) {
                rowSum += this[it, colIndex]
            }
            rowSum / cols
        })
    }

    /**
     * Calculate the mean value of each column.
     *
     * @return row vector of each mean.
     */
    fun columnWiseMean(): ComplexRowVector {
        return ComplexRowVector(cols, Array(cols) {
            var colSum = 0.0.R
            for (rowIndex in 0 until rows) {
                colSum += this[rowIndex, it]
            }
            colSum / rows
        })
    }

    /**
     * Same as [ComplexTensor.map] but returns [ComplexMatrix].
     *
     * @param lambda mappin function.
     * @return a newly mapped matrix.
     */
    override fun map(lambda: (e: ComplexDouble) -> ComplexDouble): ComplexMatrix {
        return ComplexMatrix(rows, cols, Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            lambda(this[rowIndex, colIndex])
        })
    }

    /**
     * Reshape this matrix. From the current shape, it is possible to estimate the `-1` part among the parameters.
     * One or less `-1` is allowed.
     *
     * This function does the same thing as [Tensor.reshape] (intArrayOf([newRows], [newCols]))
     * except for that it returns [Matrix].
     *
     * @param newRows a new number of rows.
     * @param newCols a new number of columns.
     * @return a new matrix with [newRows] rows and [newCols] columns.
     */
    fun reshape(newRows: Int, newCols: Int): ComplexMatrix {
        return when {
            newRows >= 0 && newCols >= 0 -> {
                if (newRows * newCols != rows * cols) throw IllegalArgumentException("ComplexMatrix.reshape: Invalid shape")
                else ComplexMatrix(newRows, newCols, data)
            }
            newRows == -1 && newCols > 0 -> {
                if ((rows * cols) % newCols != 0) throw IllegalArgumentException("ComplexMatrix.reshape: Invalid shape")
                else ComplexMatrix((rows * cols) / newCols, newCols, data)
            }
            newRows > 0 && newCols == -1 -> {
                if ((rows * cols) % newRows != 0) throw IllegalArgumentException("ComplexMatrix.reshape: Invalid shape")
                else ComplexMatrix(newRows, (rows * cols) / newRows, data)
            }
            else -> throw IllegalArgumentException("ComplexMatrix.reshape: Invalid shape")
        }
    }

    /**
     * Downcast to [ComplexRowVector] class if possible; i.e. it has only one row.
     *
     * @return downcast vector.
     */
    fun toComplexRowVector(): ComplexRowVector {
        if (rows != 1) throw IllegalStateException("ComplexMatrix.toRowVector: Cannot downcast to RowVector")
        return ComplexRowVector(cols, data)
    }

    /**
     * Downcast to [ComplexColumnVector] class if possible; i.e. it has only one column.
     *
     * @return downcast vector.
     */
    fun toComplexColVector(): ComplexColumnVector {
        if (cols != 1) throw IllegalStateException("ComplexMatrix.toColVector: Cannot downcast to ColumnVector")
        return ComplexColumnVector(rows, data)
    }

    /**
     * Same as [ComplexTensor.copy] but returns [ComplexMatrix].
     *
     * @return a new [ComplexMatrix] instance with the same shape and data.
     */
    override fun copy() = ComplexMatrix(rows, cols, data.copyOf())

    /**
     * Calculate the trace of this matrix.
     *
     * @return the trace of this matrix.
     */
    fun trace(): ComplexDouble {
        if (rows != cols) throw IllegalStateException("ComplexMatrix.trace: only for square matrices")
        return Array(rows) { this[it, it] }.sum()
    }

    /**
     * Calculate the [n]th power of this matrix.
     *
     * @param n multiplier
     * @return [n]th power of this matrix.
     */
    fun pow(n: Int): ComplexMatrix {
        if (rows != cols) throw IllegalStateException("ComplexMatrix.pow: only for square matrices")
        if (n < 0) throw IllegalArgumentException("ComplexMatrix.pow: only available for non-negative integer")
        var mat = identityMatrix(rows)
        repeat(n) { mat *= this}
        return mat
    }

    companion object {
        /**
         * Make an identity matrix of order [n].
         *
         * @param n
         * @return an identity matrix of order [n].
         */
        fun identityMatrix(n: Int): ComplexMatrix {
            val newData = Array(n * n) {
                val rowIndex = it / n
                val colIndex = it % n
                if (rowIndex == colIndex) 1.0.R else 0.0.R
            }
            return ComplexMatrix(n, n, newData)
        }

        /**
         * Make a zero matrix of shape [m] * [n].
         *
         * @param m number of rows.
         * @param n number of columns.
         * @return a zero matrix.
         */
        fun zeros(m: Int, n: Int): ComplexMatrix {
            return if (m < 1 || n < 1) throw IllegalArgumentException("ComplexMatrix.zeros: n, m must be positive integers")
            else ComplexMatrix(m, n, Array(m * n) { 0.0.R })
        }

        /**
         * Make a matrix of shape [m] * [n] in which all elements are 1
         *
         * @param m number of rows.
         * @param n number of columns.
         * @return a matrix in which all elements are 1
         */
        fun ones(m: Int, n: Int): ComplexMatrix {
            return if (m < 1 || n < 1) throw IllegalArgumentException("ComplexMatrix.ones: n, m must be positive integers")
            else ComplexMatrix(m, n, Array(m * n) { 1.0.R })
        }
    }
}
