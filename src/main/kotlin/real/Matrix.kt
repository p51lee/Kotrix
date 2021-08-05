package real

import complex.ComplexMatrix
import operations.determinant
import utils.R
import kotlin.math.cos
import kotlin.math.pow
import kotlin.math.sin

/**
 * Represents real matrix. [Matrix] class is a subclass of [Tensor] class.
 *
 * @property rows number of rows.
 * @property cols number of columns.
 * @constructor Creates a new matrix. If [data] is not given, it will generate a zero matrix.
 *
 * @param data must fit into the shape of this matrix.
 */
open class Matrix(val rows: Int, val cols: Int, data: DoubleArray = DoubleArray(rows * cols) { 0.0 }): Tensor(intArrayOf(rows, cols), data) {
    /**
     * data can be [LongArray].
     */
    constructor(rows2: Int, cols2: Int, data2: LongArray) : this(
        rows2, cols2, DoubleArray(rows2 * cols2) { data2[it].toDouble() }
    )
    /**
     * data can be [FloatArray].
     */
    constructor(rows2: Int, cols2: Int, data2: FloatArray) : this(
        rows2, cols2, DoubleArray(rows2 * cols2) { data2[it].toDouble() }
    )
    /**
     * data can be [IntArray].
     */
    constructor(rows2: Int, cols2: Int, data2: IntArray) : this(
        rows2, cols2, DoubleArray(rows2 * cols2) { data2[it].toDouble() }
    )
    /**
     * It is possible to set a value using lambda function.
     */
    constructor(rows2: Int, cols2: Int, lambda: (i: Int, j: Int) -> Number) : this(rows2, cols2,
        DoubleArray(rows2 * cols2) {
            val rowIndex = it / cols2
            val colIndex = it % cols2
            lambda(rowIndex, colIndex).toDouble()
        })

    operator fun get(rowIndex: Int, colIndex: Int) : Double {
        if (rowIndex < 0 || colIndex < 0 || rowIndex >= rows || colIndex >= cols) {
            throw IllegalArgumentException("Matrix.get: Index out of bound")
        } else {
            return data[rowIndex * cols + colIndex]
        }
    }

    operator fun set(rowIndex: Int, colIndex: Int, value: Number) {
        if (rowIndex < 0 || colIndex < 0 || rowIndex >= rows || colIndex >= cols) {
            throw IllegalArgumentException("Matrix.set: Index out of bound")
        } else {
            data[rowIndex * cols + colIndex] = value.toDouble()
        }
    }

    override operator fun unaryPlus() = this

    override operator fun unaryMinus(): Matrix {
        return Matrix(rows, cols, DoubleArray(rows * cols) {- data[it]})
    }

    operator fun plus(other: Matrix): Matrix {
        return if (rows != other.rows || cols != other.cols) {
            throw IllegalArgumentException("Matrix.plus: Two matrices should have the same shape.")
        } else {
            val newData = DoubleArray(rows * cols) {
                data[it] + other.data[it]
            }
            Matrix(rows, cols, newData)
        }
    }

    operator fun minus(other: Matrix): Matrix {
        return if (rows != other.rows || cols != other.cols) {
            throw IllegalArgumentException("Matrix.minus: Two matrices should have the same shape.")
        } else {
            val newData = DoubleArray(rows * cols) {
                data[it] - other.data[it]
            }
            Matrix(rows, cols, newData)
        }
    }

    open operator fun times(other: Matrix): Matrix {
        return if (cols != other.rows) {
            throw IllegalArgumentException("Matrix.times: Illegal Matrix multiplication.")
        } else {
            val newData = DoubleArray(rows * other.cols) {
                val rowIndex = it / other.cols
                val colIndex = it % other.cols
                var sum = 0.0
                for (i in 0 until this.cols) {
                    sum += this[rowIndex, i] * other[i, colIndex]
                }
                sum
            }
            Matrix(rows, other.cols, newData)
        }
    }

    operator fun times(other: ColumnVector): ColumnVector {
        return if (cols != other.length) {
            throw IllegalArgumentException("Matrix.times: Illegal Matrix multiplication.")
        } else {
            val newData = DoubleArray(rows * 1) {
                var sum = 0.0
                for (i in 0 until this.cols) {
                    sum += this[it, i] * other[i]
                }
                sum
            }
            ColumnVector(rows, newData)
        }
    }

    override operator fun times(other: Number): Matrix {
        val newData = DoubleArray(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            other.toDouble() * this[rowIndex, colIndex]
        }
        return Matrix(rows, cols, newData)
    }

    override operator fun div(other: Number): Matrix {
        val newData = DoubleArray(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            this[rowIndex, colIndex] / other.toDouble()
        }
        return Matrix(rows, cols, newData)
    }

    /**
     * Transpose this Matrix.
     *
     * @return transposed matrix.
     */
    open fun transpose(): Matrix {
        val newData = DoubleArray(rows * cols) {
            val transposedRowIndex = it / rows
            val transposedColIndex = it % rows
            this[transposedColIndex, transposedRowIndex]
        }
        return Matrix(cols, rows, newData)
    }

    /**
     * Calculate adjoint matrix of this matrix.
     *
     * @return adjoint matrix.
     */
    fun adjoint() : Matrix {
        if (rows != cols) throw IllegalArgumentException("Matrix.adjointMatrix: Only available for square matrices")
        val newData = DoubleArray(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            val sign = (-1.0).pow(rowIndex + colIndex)
            val cofactorDet = this.minorMatrix(rowIndex, colIndex).determinant()
            sign * cofactorDet
        }
        return Matrix(rows, cols, newData).transpose()
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
    open fun getSubmatrix(rowIndexStart: Int, rowIndexEnd: Int, colIndexStart: Int, colIndexEnd: Int): Matrix {
        return if (rowIndexStart < 0 || colIndexStart < 0 || rowIndexStart >= rowIndexEnd || colIndexStart >= colIndexEnd
            || rowIndexEnd > rows || colIndexEnd > cols) {
            throw IllegalArgumentException("Matrix.Submatrix: Index out of bound")
        } else {
            val newRows = rowIndexEnd - rowIndexStart
            val newCols = colIndexEnd - colIndexStart
            val newData = DoubleArray(newRows * newCols) {
                val newRowIndex = it / newCols
                val newColIndex = it % newCols
                this[rowIndexStart + newRowIndex, colIndexStart + newColIndex]
            }
            Matrix(newRows, newCols, newData)
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
    open fun setSubmatrix(rowIndexStart: Int, rowIndexEnd: Int, colIndexStart: Int, colIndexEnd: Int, other: Matrix) {
        val newRows = rowIndexEnd - rowIndexStart
        val newCols = colIndexEnd - colIndexStart
        if (rowIndexStart < 0 || colIndexStart < 0 || rowIndexStart >= rowIndexEnd || colIndexStart >= colIndexEnd
            || rowIndexEnd > rows || colIndexEnd > cols || newRows != other.rows || newCols != other.cols) {
            throw IllegalArgumentException("Matrix.Submatrix: Index out of bound")
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
    fun minorMatrix(rowIndex: Int, colIndex: Int) : Matrix {
        return if (rows < 2 || cols < 2 || rowIndex >= rows || colIndex >= cols) {
            throw IllegalArgumentException("Matrix.cofactorMatrix: Index out of bound")
        } else {
            val newData = DoubleArray((rows - 1) * (cols - 1)) {
                var cofactorRowIndex = it / (cols - 1)
                var cofactorColIndex = it % (cols - 1)
                if (cofactorRowIndex >= rowIndex) cofactorRowIndex += 1
                if (cofactorColIndex >= colIndex) cofactorColIndex += 1
                this[cofactorRowIndex, cofactorColIndex]
            }
            Matrix(rows - 1, cols - 1, newData)
        }
    }

    /**
     * Do a row switching transformation; exchange [rowIndex1]th row and [rowIndex2]th row.
     *
     * @param rowIndex1
     * @param rowIndex2
     * @return new transformed matrix.
     */
    fun switchRow(rowIndex1: Int, rowIndex2: Int): Matrix {
        return if (rowIndex1 < 0 || rowIndex2 < 0 || rowIndex1 >= rows || rowIndex2 >= rows) {
            throw IllegalArgumentException("Matrix.switchRow: Index out of bound")
        } else if (rowIndex1 == rowIndex2) {
            this
        } else {
            val newData = DoubleArray(rows * cols) {
                val newRowIndex = it / cols
                val newColIndex = it % cols
                when (newRowIndex) {
                    rowIndex1 -> this[rowIndex2, newColIndex]
                    rowIndex2 -> this[rowIndex1, newColIndex]
                    else -> this[newRowIndex, newColIndex]
                }
            }
            Matrix(rows, cols, newData)
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
    fun addRow(srcRowIndex: Int, dstRowIndex: Int, fraction: Number): Matrix {
        return if (srcRowIndex < 0 || dstRowIndex < 0 || srcRowIndex >= rows || dstRowIndex >= rows) {
            throw IllegalArgumentException("Matrix.addRow: Index out of bound")
        } else if (srcRowIndex == dstRowIndex) {
            throw IllegalArgumentException("Matrix.addRow: srcRow and dstRow must be different")
        } else {
            val newData = DoubleArray(rows * cols) {
                val newRowIndex = it / cols
                val newColIndex = it % cols
                when (newRowIndex) {
                    dstRowIndex -> this[dstRowIndex, newColIndex] + fraction.toDouble() * this[srcRowIndex, newColIndex]
                    else -> this[newRowIndex, newColIndex]
                }
            }
            Matrix(rows, cols, newData)
        }
    }

    /**
     * Concatenate to other matrix.
     *
     * @param other matrix to be concatenated.
     * @param concatDim dimension to which other matrix concatenate.
     * @return concatenated matrix.
     */
    fun concat(other: Matrix, concatDim: Int) : Matrix {
        return when (concatDim) {
            0 -> {
                if (cols != other.cols) throw IllegalArgumentException("Matrix.concat: number of columns does not match")
                val newRows = rows + other.rows
                val newCols = cols
                val newData = DoubleArray(newRows * newCols) {
                    val newRowIndex = it / newCols
                    val newColIndex = it % newCols
                    if (newRowIndex < rows) {
                        this[newRowIndex, newColIndex]
                    } else {
                        other[newRowIndex - rows, newColIndex]
                    }
                }
                Matrix(newRows, newCols, newData)
            }
            1 -> {
                if (rows != other.rows) throw IllegalArgumentException("Matrix.concat: number of rows does not match")
                val newRows = rows
                val newCols = cols + other.cols
                val newData = DoubleArray(newRows * newCols) {
                    val newRowIndex = it / newCols
                    val newColIndex = it % newCols
                    if (newColIndex < cols) {
                        this[newRowIndex, newColIndex]
                    } else {
                        other[newRowIndex, newColIndex - cols]
                    }
                }
                Matrix(newRows, newCols, newData)
            }
            else -> throw IllegalArgumentException("Matrix.concat: dim must be 0 or 1")
        }
    }

    /**
     * Calculate the squared norm of each column vector.
     *
     * @return row vector of each squared norm.
     */
    fun colVecNormSq(): RowVector {
        val newData = DoubleArray(1 * cols) {
            var norm = 0.0
            for (rowIndex in 0 until rows) {
                norm += this[rowIndex, it].pow(2)
            }
            norm
        }
        return RowVector(cols, newData)
    }

    /**
     * Calculate the squared norm of each row vector.
     *
     * @return column vector of each squared norm.
     */
    fun rowVecNormSq(): ColumnVector {
        val newData = DoubleArray(rows * 1) {
            var norm = 0.0
            for (colIndex in 0 until cols) {
                norm += this[it, colIndex].pow(2)
            }
            norm
        }
        return ColumnVector(rows, newData)
    }

    /**
     * Calculate the sum of all values in this matrix.
     *
     * @return the sum.
     */
    fun sum(): Double {
        var sum = 0.0
        data.forEach { sum += it }
        return sum
    }

    /**
     * Element-wise multiplication.
     *
     * @param other must have the same shape as this matrix.
     * @return multiplied matrix.
     */
    fun eltwiseMul(other: Matrix): Matrix {
        if (rows != other.rows || cols != other.cols)
            throw IllegalArgumentException("Matrix.eltwiseMul: Both operands must have the same shape")
        return Matrix(rows, cols, DoubleArray(rows * cols) {
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
    fun rowWiseMean(): ColumnVector {
        return ColumnVector(rows, DoubleArray(rows) {
            var rowSum = 0.0
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
    fun columnWiseMean(): RowVector {
        return RowVector(cols, DoubleArray(cols) {
            var colSum = 0.0
            for (rowIndex in 0 until rows) {
                colSum += this[rowIndex, it]
            }
            colSum / rows
        })
    }

    /**
     * Same as [Tensor.map] but returns [Matrix].
     *
     * @param lambda mapping function.
     * @return a newly mapped matrix.
     */
    override fun map(lambda: (e: Double) -> Number): Matrix {
        return Matrix(rows, cols, DoubleArray(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            lambda(this[rowIndex, colIndex]).toDouble()
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
    fun reshape(newRows: Int, newCols: Int): Matrix {
        return when {
            newRows >= 0 && newCols >= 0 -> {
                if (newRows * newCols != rows * cols) throw IllegalArgumentException("Matrix.reshape: Invalid shape")
                else Matrix(newRows, newCols, data)
            }
            newRows == -1 && newCols > 0 -> {
                if ((rows * cols) % newCols != 0) throw IllegalArgumentException("Matrix.reshape: Invalid shape")
                else Matrix((rows * cols) / newCols, newCols, data)
            }
            newRows > 0 && newCols == -1 -> {
                if ((rows * cols) % newRows != 0) throw IllegalArgumentException("Matrix.reshape: Invalid shape")
                else Matrix(newRows, (rows * cols) / newRows, data)
            }
            else -> throw IllegalArgumentException("Matrix.reshape: Invalid shape")
        }
    }

    /**
     * Downcast to [RowVector] class if possible; i.e. it has only one row.
     *
     * @return downcast vector.
     */
    fun toRowVector(): RowVector {
        if (rows != 1) throw IllegalStateException("Matrix.toRowVector: Cannot downcast to RowVector")
        return RowVector(cols, data)
    }

    /**
     * Downcast to [ColumnVector] class if possible; i.e. it has only one column.
     *
     * @return downcast vector.
     */
    fun toColVector(): ColumnVector {
        if (cols != 1) throw IllegalStateException("Matrix.toColVector: Cannot downcast to ColumnVector")
        return ColumnVector(rows, data)
    }

    /**
     * Same as [Tensor.toComplex] but returns [ComplexMatrix].
     *
     * @return complex matrix.
     */
    override fun toComplex(): ComplexMatrix {
        return ComplexMatrix(rows, cols, Array(rows * cols) { data[it].R })
    }

    /**
     * Same as [Tensor.copy] but returns [Matrix]
     *
     * @return a new [Matrix] instance with the same shape and data.
     */
    override fun copy() = Matrix(rows, cols, data.copyOf())

    /**
     * Calculate the trace of this matrix.
     *
     * @return the trace of this matrix.
     */
    fun trace(): Double {
        if (rows != cols) throw IllegalStateException("Matrix.trace: only for square matrices")
        return DoubleArray(rows) { this[it, it] }.sum()
    }

    /**
     * Calculate the [n]th power of this matrix.
     *
     * @param n multiplier
     * @return [n]th power of this matrix.
     */
    fun pow(n: Int): Matrix {
        if (rows != cols) throw IllegalStateException("Matrix.pow: only for square matrices")
        if (n < 0) throw IllegalArgumentException("Matrix.pow: only available for non-negative integer")
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
        fun identityMatrix(n: Int): Matrix {
            val newData = DoubleArray(n * n) {
                val rowIndex = it / n
                val colIndex = it % n
                if (rowIndex == colIndex) 1.0 else 0.0
            }
            return Matrix(n, n, newData)
        }

        /**
         * Make a zero matrix of shape [m] * [n].
         *
         * @param m number of rows.
         * @param n number of columns.
         * @return a zero matrix.
         */
        fun zeros(m: Int, n: Int): Matrix {
            return if (m < 1 || n < 1) throw IllegalArgumentException("Matrix.zeros: n, m must be positive integers")
            else Matrix(m, n, DoubleArray(m * n) { 0.0 })
        }

        /**
         * Make a matrix of shape [m] * [n] in which all elements are 1
         *
         * @param m number of rows.
         * @param n number of columns.
         * @return a matrix in which all elements are 1
         */
        fun ones(m: Int, n: Int): Matrix {
            return if (m < 1 || n < 1) throw IllegalArgumentException("Matrix.ones: n, m must be positive integers")
            else Matrix(m, n, DoubleArray(m * n) { 1.0 })
        }

        /**
         * Make a 2-dimensional rotation matrix.
         *
         * @param theta rotation angle in radian.
         * @return 2-dimensional rotation matrix.
         */
        fun rotationMatrix2d(theta: Double): Matrix {
            return Matrix(2, 2, doubleArrayOf(
                cos(theta), -sin(theta),
                sin(theta), cos(theta)
            ))
        }

        /**
         * Make a 3-dimensional rotation matrix about the x-axis.
         *
         * @param theta rotation angle in radian.
         * @return 3-dimensional rotation matrix.
         */
        fun rotationMatrix3dX(theta: Double): Matrix {
            return Matrix(3, 3, doubleArrayOf(
                1.0,        0.0,            0.0,
                0.0,        cos(theta),     -sin(theta),
                0.0,        sin(theta),     cos(theta)
            ))
        }

        /**
         * Make a 3-dimensional rotation matrix about the y-axis.
         *
         * @param theta rotation angle in radian.
         * @return 3-dimensional rotation matrix.
         */
        fun rotationMatrix3dY(theta: Double): Matrix {
            return Matrix(3, 3, doubleArrayOf(
                cos(theta),     0.0,        sin(theta),
                0.0,            1.0,        0.0,
                -sin(theta),    0.0,        cos(theta)
            ))
        }

        /**
         * Make a 3-dimensional rotation matrix about the z-axis.
         *
         * @param theta rotation angle in radian.
         * @return 3-dimensional rotation matrix.
         */
        fun rotationMatrix3dZ(theta: Double): Matrix {
            return Matrix(3, 3, doubleArrayOf(
                cos(theta),     -sin(theta),    0.0,
                sin(theta),     cos(theta),     0.0,
                0.0,            0.0,            1.0
            ))
        }

        /**
         * Make a 3-dimensional rotation matrix of given Euler angles.
         *
         * Its sequence of rotation axes is z-x-z.
         *
         * @param alpha first Euler angle.
         * @param beta second Euler angle.
         * @param gamma third Euler angle.
         * @return
         */
        fun eulerRotationMatrix3d(alpha: Double, beta: Double, gamma: Double): Matrix {
            return rotationMatrix3dZ(alpha) * rotationMatrix3dX(beta) * rotationMatrix3dZ(gamma)
        }

        /**
         * Make an elementary matrix which represents row-switching transformation.
         *
         * @param n order of the matrix.
         * @param firstIndex index of the first row to be exchanged.
         * @param secondIndex index of the second row to be exchaged.
         * @return an elementary matrix.
         */
        fun rowSwitchingMatrix(n: Int, firstIndex: Int, secondIndex: Int): Matrix {
            if (firstIndex >= n || secondIndex >= n) throw IllegalArgumentException("Matrix.rowSwitchingMatrix: Index out of bound")
            return identityMatrix(n).switchRow(firstIndex, secondIndex)
        }
    }
}