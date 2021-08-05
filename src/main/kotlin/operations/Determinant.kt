package operations

import complex.ComplexDouble
import complex.ComplexMatrix
import real.Matrix
import utils.R
import utils.times
import utils.div

/**
 * Calculates a determiant using LU decomposition through recursion, O(n^3).
 *
 * @return determinant of a matrix.
 */
fun Matrix.determinant(): Double {
    if (rows != cols) throw IllegalArgumentException("Matrix.determinant: Only available for square matrices")
    when (rows) {
        1 -> return this[0, 0]
        2 -> return this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0]
        else -> {
            var sign = 1
            var switchIndex = 0
            val firstColumn = DoubleArray(rows) { this[it, 0] }
            var a = 0.0

            for (elem in firstColumn) {
                if (elem != 0.0) {
                    a = elem
                    break
                } else {
                    switchIndex += 1
                }
            }
            return if (a == 0.0) 0.0 // 첫 번째 열이 모두 0이다.
            else {
                if (switchIndex != 0) sign = -1
                val matP1A = switchRow(0, switchIndex)
                val v = matP1A.getSubmatrix(1, rows, 0, 1)
                val wT = matP1A.getSubmatrix(0, 1, 1, cols)
                val c = 1 / a
                val matAPrime = matP1A.getSubmatrix(1, rows, 1, cols)
                sign * a * (matAPrime - (v * wT) * c).determinant()
            }
        }
    }
}

/**
 * Calculates a determiant using LU decomposition through recursion, O(n^3).
 *
 * @return determinant of a matrix.
 */
fun ComplexMatrix.determinant(): ComplexDouble {
    // using LU decomposition through recursion, O(n^3)
    if (rows != cols) throw IllegalArgumentException("ComplexMatrix.determinant: Only available for square matrices")
    when (rows) {
        1 -> return this[0, 0]
        2 -> return this[0, 0] * this[1, 1] - this[0, 1] * this[1, 0]
        else -> {
            var sign = 1
            var switchIndex = 0
            val firstColumn = Array(rows) { this[it, 0] }
            var a = 0.0.R

            for (elem in firstColumn) {
                if (elem != 0.0.R) {
                    a = elem
                    break
                } else {
                    sign *= -1
                    switchIndex += 1
                }
            }
            return if (a == 0.0.R) 0.0.R // 첫 번째 열이 모두 0이다.
            else {
                val matP1A = switchRow(0, switchIndex)
                val v = matP1A.getSubmatrix(1, rows, 0, 1)
                val wT = matP1A.getSubmatrix(0, 1, 1, cols)
                val c = 1 / a
                val matAPrime = matP1A.getSubmatrix(1, rows, 1, cols)
                sign * a * (matAPrime - (v * wT) * c).determinant()
            }
        }
    }
}