package operations

import complex.ComplexMatrix
import real.Matrix
import utils.R
import utils.times
import utils.div

/**
 * PLU decomposition of a matrix.
 *
 * @return an array of { permutation matrix, lower triangle matrix, upper triangle matrix }.
 */
fun Matrix.plu(): Array<Matrix> { // PLU
    if (rows != cols) throw IllegalArgumentException("Matrix.plu: Only available for square matrices")
    return when (rows) {
        1 -> arrayOf(Matrix.identityMatrix(1), Matrix.identityMatrix(1), this)
        else -> {
            var switchIndex = 0
            val firstColumn = DoubleArray(rows) { this[it, 0] }
            var a = 0.0

            for (elem in firstColumn) {
                if (elem != 0.0) {
                    a = elem; break
                } else switchIndex += 1
            }

            val matP1 = Matrix.rowSwitchingMatrix(rows, 0, switchIndex)
            val matP1A = switchRow(0, switchIndex)
            val v = matP1A.getSubmatrix(1, rows, 0, 1)
            val wT = matP1A.getSubmatrix(0, 1, 1, cols)
            val c = if (a != 0.0) 1 / a else 0.0
            val matAPrime = matP1A.getSubmatrix(1, rows, 1, cols)

            val pLUPrime = (matAPrime - (v * wT) * c).plu()
            val cvPrime = c * pLUPrime[0] * v

            val matPprime = Matrix.identityMatrix(rows)
            matPprime.setSubmatrix(1, rows, 1, cols, pLUPrime[0])
            val matP = matPprime * matP1

            val matL = Matrix.identityMatrix(rows)
            matL.setSubmatrix(1, rows, 1, cols, pLUPrime[1])
            matL.setSubmatrix(1, rows, 0, 1, cvPrime)

            val matU = Matrix.identityMatrix(rows)
            matU[0, 0] = a
            matU.setSubmatrix(0, 1, 1, cols, wT)
            matU.setSubmatrix(1, rows, 1, cols, pLUPrime[2])

            arrayOf(matP, matL, matU)
        }
    }
}

/**
 * PLU decomposition of a matrix.
 *
 * @return an array of { permutation matrix, lower triangle matrix, upper triangle matrix }.
 */
fun ComplexMatrix.plu(): Array<ComplexMatrix> { // PLU
    if (rows != cols) throw IllegalArgumentException("ComplexMatrix.plu: Only available for square matrices")
    return when (rows) {
        1 -> arrayOf(Matrix.identityMatrix(1).toComplex(), Matrix.identityMatrix(1).toComplex(), this)
        else -> {
            var switchIndex = 0
            val firstColumn = Array(rows) { this[it, 0] }
            var a = 0.0.R

            for (elem in firstColumn) {
                if (elem != 0.0.R) {
                    a = elem; break
                } else switchIndex += 1
            }

            val matP1 = Matrix.rowSwitchingMatrix(rows, 0, switchIndex).toComplex()
            val matP1A = switchRow(0, switchIndex)
            val v = matP1A.getSubmatrix(1, rows, 0, 1)
            val wT = matP1A.getSubmatrix(0, 1, 1, cols)
            val c = if (a != 0.0.R) 1 / a else 0.0.R
            val matAPrime = matP1A.getSubmatrix(1, rows, 1, cols)

            val pLUPrime = (matAPrime - (v * wT) * c).plu()
            val cvPrime = c * pLUPrime[0] * v

            val matPprime = Matrix.identityMatrix(rows).toComplex()
            matPprime.setSubmatrix(1, rows, 1, cols, pLUPrime[0])
            val matP = matPprime * matP1

            val matL = Matrix.identityMatrix(rows).toComplex()
            matL.setSubmatrix(1, rows, 1, cols, pLUPrime[1])
            matL.setSubmatrix(1, rows, 0, 1, cvPrime)

            val matU = Matrix.identityMatrix(rows).toComplex()
            matU[0, 0] = a
            matU.setSubmatrix(0, 1, 1, cols, wT)
            matU.setSubmatrix(1, rows, 1, cols, pLUPrime[2])

            arrayOf(matP, matL, matU)
        }
    }
}