package operations

import complex.ComplexMatrix
import real.Matrix
import utils.R
import kotlin.math.pow

fun Matrix.inverse() : Matrix {
    if (rows != cols) throw IllegalArgumentException("Matrix.inverseMatrix: Only available for square matrices")
    val det = this.determinant()
    return if (det == 0.0) Matrix.identityMatrix(rows)
    else this.adjoint() * (det.pow(-1))
}

fun ComplexMatrix.inverse() : ComplexMatrix {
    if (rows != cols) throw IllegalArgumentException("ComplexMatrix.inverseMatrix: Only available for square matrices")
    val det = this.determinant()
    return if (det == 0.0.R) ComplexMatrix.identityMatrix(rows)
    else this.adjointMatrix() / det
}