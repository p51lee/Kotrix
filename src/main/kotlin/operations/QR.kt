package operations

import real.ColumnVector
import real.Matrix
import utils.pseudoEquals

/**
 * Calculates QR decomposition of square matrix.
 *
 * @return an array of { orthogonal matrix, upper triangular matrix }.
 */
fun Matrix.qr(): Array<Matrix> {
    if (rows != cols) throw IllegalArgumentException("Matrix.qr: Only available for square matrices")

    lateinit var u: ColumnVector
    val unitVectors = arrayListOf<ColumnVector>()

    for (colIndex in 0 until cols) {
        val a = this.getSubmatrix(0, rows, colIndex, colIndex + 1).toColVector()
        u = a
        for (eIndex in 0 until colIndex) {
            u -= a.proj(unitVectors[eIndex])
        }
        if (pseudoEquals(u.frobeniusNormSquared(), 0.0)) {
            unitVectors.add(ColumnVector(rows))
        } else {
            unitVectors.add(u.normalize())
        }
    }
    val matQ = Matrix.identityMatrix(rows)
    for (colIndex in  0 until cols) {
        matQ.setSubmatrix(0, rows, colIndex, colIndex + 1, unitVectors[colIndex])
    }
    val matR = matQ.transpose() * this
    return arrayOf(matQ, matR)
}
