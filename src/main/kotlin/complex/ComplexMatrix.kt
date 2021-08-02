package complex

import operations.determinant
import real.ColumnVector
import utils.R
import utils.sum
import kotlin.math.pow
import utils.times

open class ComplexMatrix (val rows: Int, val cols: Int, data: Array<ComplexDouble> = Array(rows * cols) { 0.0.R }) :
    ComplexTensor(intArrayOf(rows, cols), data) {

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

    open fun transpose(): ComplexMatrix {
        val newData = Array(rows * cols) {
            val transposedRowIndex = it / rows
            val transposedColIndex = it % rows
            this[transposedColIndex, transposedRowIndex]
        }
        return ComplexMatrix(cols, rows, newData)
    }

    open fun conjTrans(): ComplexMatrix {
        val newData = Array(rows * cols) {
            val transposedRowIndex = it / rows
            val transposedColIndex = it % rows
            this[transposedColIndex, transposedRowIndex].conj()
        }
        return ComplexMatrix(cols, rows, newData)
    }

    fun adjointMatrix() : ComplexMatrix {
        if (rows != cols) throw IllegalArgumentException("ComplexMatrix.adjointMatrix: Only available for square matrices")
        val newData = Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            val sign = (-1.0).pow(rowIndex + colIndex)
            val cofactorDet = this.cofactorMatrix(rowIndex, colIndex).determinant()
            sign * cofactorDet
        }
        return ComplexMatrix(rows, cols, newData).transpose()
    }

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

    fun cofactorMatrix(rowIndex: Int, colIndex: Int) : ComplexMatrix {
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

    fun addRow(srcRowIndex: Int, dstRowIndex: Int, fraction: Double): ComplexMatrix {
        return if (srcRowIndex < 0 || dstRowIndex < 0 || srcRowIndex >= rows || dstRowIndex >= rows) {
            throw IllegalArgumentException("ComplexMatrix.addRow: Index out of bound")
        } else if (srcRowIndex == dstRowIndex) {
            throw IllegalArgumentException("ComplexMatrix.addRow: srcRow and dstRow must be different")
        } else {
            val newData = Array(rows * cols) {
                val newRowIndex = it / cols
                val newColIndex = it % cols
                when (newRowIndex) {
                    dstRowIndex -> this[dstRowIndex, newColIndex] + fraction * this[srcRowIndex, newColIndex]
                    else -> this[newRowIndex, newColIndex]
                }
            }
            ComplexMatrix(rows, cols, newData)
        }
    }

    fun concat(other: ComplexMatrix, dim: Int) : ComplexMatrix {
        return when (dim) {
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

    fun sum(): ComplexDouble {
        var sum = 0.0.R
        data.forEach { sum += it }
        return sum
    }

    fun eltwiseMul(other: ComplexMatrix): ComplexMatrix {
        if (rows != other.rows || cols != other.cols)
            throw IllegalArgumentException("ComplexMatrix.eltwiseMul: Both operands must have the same shape")
        return ComplexMatrix(rows, cols, Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            this[rowIndex, colIndex] * other[rowIndex, colIndex]
        })
    }

    fun rowWiseMean(): ComplexColumnVector {
        return ComplexColumnVector(rows, Array(rows) {
            var rowSum = 0.0.R
            for (colIndex in 0 until cols) {
                rowSum += this[it, colIndex]
            }
            rowSum / cols
        })
    }

    fun columnWiseMean(): ComplexRowVector {
        return ComplexRowVector(cols, Array(cols) {
            var colSum = 0.0.R
            for (rowIndex in 0 until rows) {
                colSum += this[rowIndex, it]
            }
            colSum / rows
        })
    }

    override fun map(lambda: (e: ComplexDouble) -> ComplexDouble): ComplexMatrix {
        return ComplexMatrix(rows, cols, Array(rows * cols) {
            val rowIndex = it / cols
            val colIndex = it % cols
            lambda(this[rowIndex, colIndex])
        })
    }

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

    fun toComplexRowVector(): ComplexRowVector {
        if (rows != 1) throw IllegalStateException("ComplexMatrix.toRowVector: Cannot downcast to RowVector")
        return ComplexRowVector(cols, data)
    }

    fun toComplexColVector(): ComplexColumnVector {
        if (cols != 1) throw IllegalStateException("ComplexMatrix.toColVector: Cannot downcast to ColumnVector")
        return ComplexColumnVector(rows, data)
    }

    override fun copy() = ComplexMatrix(rows, cols, data.copyOf())

    fun trace(): ComplexDouble {
        if (rows != cols) throw IllegalStateException("ComplexMatrix.trace: only for square matrices")
        return Array(rows) { this[it, it] }.sum()
    }

    fun pow(n: Int): ComplexMatrix {
        if (rows != cols) throw IllegalStateException("ComplexMatrix.pow: only for square matrices")
        if (n < 0) throw IllegalArgumentException("ComplexMatrix.pow: only available for non-negative integer")
        var mat = ComplexMatrix.identityMatrix(rows)
        repeat(n) { mat *= this}
        return mat
    }

    companion object {
        fun identityMatrix(dim: Int): ComplexMatrix {
            val newData = Array(dim * dim) {
                val rowIndex = it / dim
                val colIndex = it % dim
                if (rowIndex == colIndex) 1.0.R else 0.0.R
            }
            return ComplexMatrix(dim, dim, newData)
        }

        fun zeros(n: Int, m: Int): ComplexMatrix {
            return if (n < 1 || m < 1) throw IllegalArgumentException("ComplexMatrix.zeros: n, m must be positive integers")
            else ComplexMatrix(n, m, Array(n * m) { 0.0.R })
        }

        fun ones(n: Int, m: Int): ComplexMatrix {
            return if (n < 1 || m < 1) throw IllegalArgumentException("ComplexMatrix.ones: n, m must be positive integers")
            else ComplexMatrix(n, m, Array(n * m) { 1.0.R })
        }
    }
}
