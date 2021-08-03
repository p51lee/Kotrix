package operations

import complex.ComplexColumnVector
import complex.ComplexDouble
import complex.ComplexMatrix
import complex.ComplexRowVector
import real.ColumnVector
import real.Matrix
import utils.*

private fun Matrix.calcDegeneracyMap(eigenValues: Array<ComplexDouble>): Map<ComplexDouble, Int> {
    val degeneracies = mutableMapOf<ComplexDouble, Int>()

    eigenValues.forEachIndexed { index, eigenValue ->
        for (i in index until eigenValues.lastIndex) {
            if ((eigenValue - eigenValues[i + 1]).abs() < equalityValidation)
                eigenValues[i + 1] = eigenValue
        }
    }

    eigenValues.forEach {
        if (it in degeneracies.keys) degeneracies[it] = degeneracies[it]!! + 1
        else degeneracies[it] = 1
    }
    return degeneracies.toMap().toSortedMap(compareBy<ComplexDouble> {-it.re})
}

private fun Matrix.calcCharacteristicEq(): (ComplexDouble) -> ComplexDouble {
    val n = rows
    val constantArray = DoubleArray(n + 1)
    for (m in 0..n) {
        constantArray[n - m] =
            if (m == 0) 1.0 else {
                var sum = 0.0
                for (k in 1..m) {
                    sum += constantArray[n - m + k] * (this.pow(k)).trace()
                }
                - sum / m
            }
    }
    return  { x: ComplexDouble ->
        var sum = 0.R
        constantArray.forEachIndexed { index, constant ->
            sum += x.pow(index) * constant
        }
        sum
    }
}

private fun Matrix.durandKernerMethod(f: (ComplexDouble) -> ComplexDouble): Array<ComplexDouble> {
    var p0 = 0.4.R + 0.9.I
    while (true) {
        var isRoot = false
        for (i in 1..rows) {
            if (f(p0.pow(i)) == 0.R) isRoot = true
        }
        if (!isRoot) break
        else p0 *= 2
    }
    val roots = Array(rows) { p0.pow(it + 1) }
    var diff = 1.0
    while(diff > convCheck) {
        val prevRoots = roots.copyOf()
        for (i in 0 until rows) {
            val product = prevRoots.foldIndexed(1.R) { index, acc, it ->
                if (index == i) acc else acc * (prevRoots[i] - it)
            }
            roots[i] = prevRoots[i] - f(prevRoots[i]) / product
        }
        diff = prevRoots.foldIndexed(0.0) { index, acc, it ->
            acc + (roots[index] - it).abs()
        } / prevRoots.size
    }
    return roots
}

/**
 * Solves eigenvalue problem of a square matrix and finds eigenvectors of each eigenvalue. The answer may be complex.
 *
 * @return an array of { matrix of eigenvectors, diagonal matrix of eigenvalues }.
 */
fun Matrix.eig(): Array<ComplexMatrix> {
    if (rows != cols) throw IllegalArgumentException("Matrix.evp: Only available for square matrices")

    val f = calcCharacteristicEq()

    val eigenVectors = arrayListOf<ComplexColumnVector>()
    val eigenValues = durandKernerMethod(f)
    val degeneracyMap = calcDegeneracyMap(eigenValues)
    val eigenValuesOrdered = arrayListOf<ComplexDouble>()

    degeneracyMap.forEach { (eigenValue, degeneracy) ->
        repeat (degeneracy) {
            eigenValuesOrdered.add(eigenValue)
        }
        val charMat = this.toComplex() - ComplexMatrix.identityMatrix(rows) * eigenValue
        val charMatReduced = charMat.rowEchelonForm()

        val noOverlapMat = charMatReduced.copy()
        for (colIndex in 0 until cols) {
            var nonzeroAppeared = false
            for (rowIndex in 0 until rows) {
                if (nonzeroAppeared) {
                    noOverlapMat[rowIndex, colIndex] = 0.R
                } else if (!pseudoEquals(noOverlapMat[rowIndex, colIndex].abs(), 0.0)) {
                    nonzeroAppeared = true
                }
            }
        }

        val appearanceRate = IntArray(cols)
        val canBe1Index = arrayListOf<Int>()

        for (rowIndex in 0 until rows) {
            val nonzeroElementIndices = arrayListOf<Int>()
            for (colIndex in 0 until cols) {
                if (!pseudoEquals(noOverlapMat[rowIndex, colIndex].abs(), 0.0)) {
                    if (colIndex < cols) nonzeroElementIndices.add(colIndex)
                    appearanceRate[colIndex] = 1
                }
            }
            for (i in 0 until nonzeroElementIndices.size - 1) { // 맨 마지막꺼는 안함
                canBe1Index.add(nonzeroElementIndices[i])
            }
        }
        appearanceRate.forEachIndexed { index, appearance ->
            if (appearance == 0) canBe1Index.add(index)
        }

        for (index1 in 0 until canBe1Index.size) {
            val modifiedMat = charMatReduced.copy()
            val rhsVec = ColumnVector(rows)
            var rowPointer = rows - 1
            for (index2 in 0 until canBe1Index.size) {
                modifiedMat.setSubmatrix(rowPointer, rowPointer + 1, 0, cols,
                    ComplexRowVector(cols) { if (it == canBe1Index[index2]) 1.R else 0.R }
                )
                rhsVec[rowPointer] = if (index2 != index1) 0.0 else 1.0
                rowPointer--
            }
            val eigenVector = (modifiedMat.inverse() * rhsVec).normalize()
            eigenVectors.add(eigenVector)
        }
    }

    val matD = ComplexMatrix(rows, cols) { i, j ->
        if (i == j) eigenValuesOrdered[i] else 0.R
    }
    val matV = eigenVectors.foldIndexed(eigenVectors[0] as ComplexMatrix) { i, mat, it ->
        if (i != 0) mat.concat(it, 1) else mat
    }
    return arrayOf(matV, matD)
}
