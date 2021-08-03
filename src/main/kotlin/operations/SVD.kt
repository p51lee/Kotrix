package operations

import complex.ComplexMatrix
import real.Matrix
import utils.R

/**
 * Calculate the singular value decomposition of a matrix. The result may be complex.
 *
 * @return an array of { unitary matrix(U), rectangular diagonal matrix (Sigma)) , unitary matrix (VT) }
 */
fun Matrix.svd(): Array<ComplexMatrix> {
    val matAAT = this * this.transpose()
    val matATA = this.transpose() * this
    val eigAAT = matAAT.eig()
    val eigATA = matATA.eig()

    val matSigma =
        if (rows > cols) {
            ComplexMatrix(rows, cols) { i, j ->
                if (i == j)
                    eigAAT[1][i, j].pow(0.5)
                else
                    0.R
            }
        } else {
            ComplexMatrix(rows, cols) { i, j ->
                if (i == j)
                    eigATA[1][i, j].pow(0.5)
                else
                    0.R
            }
        }
    val matU = eigAAT[0]
    val matV = eigATA[0]
    return arrayOf(matU, matSigma, matV.transpose())
}