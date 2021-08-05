package operations

import complex.ComplexMatrix
import real.Matrix
import utils.R
import utils.pseudoEquals
import kotlin.math.abs

/**
 * Calculate reduced row echelon form using Gaussian elimination.
 *
 * @return matrix in reduced row echelon form.
 */
fun Matrix.rowEchelonForm(): Matrix {
    var newMat = this.copy()
    var h = 0
    var k = 0

    while (h < rows && k < cols) {
        val iMax = (h until rows).maxByOrNull { abs(newMat[it, k]) }!!
        if (newMat[iMax, k] == 0.0) k++
        else {
            newMat = newMat.switchRow(h, iMax)
            for (i in (0 until h) + (h + 1 until rows)) {
                val f = newMat[i, k] / newMat[h, k]
                newMat[i, k] = 0.0
                for (j in k + 1 until cols) {
                    newMat[i, j] = newMat[i, j] - newMat[h, j] * f
                }
            }
            h++
            k++
        }
    }
    var reducedMat = Matrix(rows, cols, newMat.data.map {
        if (pseudoEquals(it, 0.0)) 0.0 else it
    }.toDoubleArray())

    val sortedMat = Matrix(rows, cols)
    var rowPointer = 0
    for (rowIndex in 0 until rows) {
        var isZeroRowVec = true
        for (colIndex in 0 until cols) {
            if (reducedMat[rowIndex, colIndex] != 0.0) {
                isZeroRowVec = false
                break
            }
        }
        if (!isZeroRowVec) {
            sortedMat.setSubmatrix(rowPointer, rowPointer + 1, 0, cols,
                reducedMat.getSubmatrix(rowIndex, rowIndex + 1, 0, cols))
            rowPointer++
        }
    }
    return sortedMat
}

/**
 * Calculate reduced row echelon form using Gaussian elimination.
 *
 * @return matrix in reduced row echelon form.
 */
fun ComplexMatrix.rowEchelonForm(): ComplexMatrix { // reduced row echelon form
    var newMat = this.copy()
    var h = 0
    var k = 0

    while (h < rows && k < cols) {
        val iMax = (h until rows).maxByOrNull { newMat[it, k].abs() }!!
        if (pseudoEquals(newMat[iMax, k] , 0.R)) k++
        else {
            newMat = newMat.switchRow(h, iMax)
            for (i in (0 until h) + (h + 1 until rows)) {
                val f = newMat[i, k] / newMat[h, k]
                newMat[i, k] = 0.R
                for (j in k + 1 until cols) {
                    newMat[i, j] = newMat[i, j] - newMat[h, j] * f
                }
            }
            h++
            k++
        }
    }
    var reducedMat = ComplexMatrix(rows, cols, newMat.data.map {
        if (pseudoEquals(it, 0.R)) 0.R else it
    }.toTypedArray())

    val sortedMat = ComplexMatrix(rows, cols)
    var rowPointer = 0
    for (rowIndex in 0 until rows) {
        var isZeroRowVec = true
        for (colIndex in 0 until cols) {
            if (!pseudoEquals(reducedMat[rowIndex, colIndex], 0.R)) {
                isZeroRowVec = false
                break
            }
        }
        if (!isZeroRowVec) {
            sortedMat.setSubmatrix(rowPointer, rowPointer + 1, 0, cols,
                reducedMat.getSubmatrix(rowIndex, rowIndex + 1, 0, cols))
            rowPointer++
        }
    }
    return sortedMat
}