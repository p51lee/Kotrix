import complex.ComplexMatrix
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import real.ColumnVector
import real.Matrix
import real.RowVector
import utils.R
import utils.times

internal class MatrixTest {
    private val mat1 = Matrix(4, 4, intArrayOf(
        4, 3, 2, 2,
        0, 1, -3, 3,
        0, -1, 3, 3,
        0, 3, 1, 1
    ))
    private val negMat1 = Matrix(4, 4, intArrayOf(
        -4, -3, -2, -2,
        0, -1, 3, -3,
        0, 1, -3, -3,
        0, -3, -1, -1
    ))
    private val twiceMat1 = Matrix(4, 4, intArrayOf(
        8, 6, 4, 4,
        0, 2, -6, 6,
        0, -2, 6, 6,
        0, 6, 2, 2
    ))
    private val squareMat1 = Matrix(4, 4, intArrayOf(
        16, 19, 7, 25,
        0, 13, -9, -3,
        0, 5, 15, 9,
        0, 5, -5, 13
    ))

    @Test
    fun get() {
        assertEquals(-1.0, mat1[2, 1], 0.00001)
    }

    @Test
    fun set() {
        val mat = Matrix(4, 4, intArrayOf(
            4, 3, 2, 2,
            0, 1, -3, 3,
            0, -1, 3, 3,
            0, 3, 1, 1
        ))
        mat[2, 1] = 20181102
        assertEquals(20181102.0, mat[2, 1],0.00001)
    }

    @Test
    operator fun unaryPlus() {
        assertEquals(mat1, +mat1)
    }

    @Test
    operator fun unaryMinus() {
        assertEquals(negMat1, -mat1)
    }

    @Test
    fun plus() {
        assertEquals(twiceMat1, mat1 + mat1)
    }

    @Test
    fun minus() {
        assertEquals(mat1, twiceMat1 - mat1)
    }

    @Test
    fun times() {
        assertEquals(twiceMat1, mat1 * 2)
        assertEquals(twiceMat1, 2 * mat1)
        assertEquals(squareMat1, mat1 * mat1)
    }

    @Test
    fun div() {
        assertEquals(mat1, twiceMat1 / 2)
    }

    @Test
    fun transpose() {
        val matT = Matrix(4, 4, intArrayOf(
            4, 0, 0, 0,
            3, 1, -1, 3,
            2, -3, 3, 1,
            2, 3, 3, 1
        ))
        assertEquals(matT, mat1.transpose())
    }

    @Test
    fun frobeniusNormSquared() {
        assertEquals(82.0, mat1.frobeniusNormSquared(), 0.00001)
    }

    @Test
    fun adjointMatrix() {
        val adjMat1 = Matrix(4, 4, intArrayOf(
            -60, 0, 18, 66,
            0, 0, 24, -72,
            0, 40, -32, -24,
            0, -40, -40, 0
        ))
        assertEquals(adjMat1, mat1.adjoint())
    }

    @Test
    fun getSubmatrix() {
        val subMat1 = Matrix(2, 2, intArrayOf(
            1, -3,
            -1, 3
        ))
        assertEquals(subMat1, mat1.getSubmatrix(1, 3, 1, 3))
    }

    @Test
    fun setSubmatrix() {
        val mat = Matrix(4, 4, intArrayOf(
            4, 3, 2, 2,
            0, 1, -3, 3,
            0, -1, 3, 3,
            0, 3, 1, 1
        ))
        val subMat = Matrix(2, 2, intArrayOf(
            1, 2,
            3, 4
        ))
        val changedMat = Matrix(4, 4, intArrayOf(
            4, 3, 2, 2,
            0, 1, 2, 3,
            0, 3, 4, 3,
            0, 3, 1, 1
        ))
        mat.setSubmatrix(1, 3, 1, 3, subMat)
        assertEquals(changedMat, mat)
    }

    @Test
    fun cofactorMatrix() {
        val cofMat1 = Matrix(3, 3, intArrayOf(
            4, 3, 2,
            0, 1, -3,
            0, -1, 3
        ))
        assertEquals(cofMat1, mat1.minorMatrix(3, 3))
    }

    @Test
    fun switchRow() {
        val newMat = Matrix(4, 4, intArrayOf(
            4, 3, 2, 2,
            0, -1, 3, 3,
            0, 1, -3, 3,
            0, 3, 1, 1
        ))
        assertEquals(newMat, mat1.switchRow(1, 2))
    }

    @Test
    fun addRow() {
        val newMat = Matrix(4, 4, intArrayOf(
            4, 3, 2, 2,
            0, -1, 3, 9,
            0, -1, 3, 3,
            0, 3, 1, 1
        ))
        assertEquals(newMat, mat1.addRow(2, 1, 2.0))
    }

    @Test
    fun concat() {
        val concatMat = Matrix(8, 4, intArrayOf(
            4, 3, 2, 2,
            0, 1, -3, 3,
            0, -1, 3, 3,
            0, 3, 1, 1,
            4, 3, 2, 2,
            0, 1, -3, 3,
            0, -1, 3, 3,
            0, 3, 1, 1
        ))
        assertEquals(concatMat, mat1.concat(mat1, 0))
    }

    @Test
    fun colVecNormSq() {
        val rowVec = RowVector(4, intArrayOf(16, 20, 23, 23))
        assertEquals(rowVec, mat1.colVecNormSq())
    }

    @Test
    fun rowVecNormSq() {
        val colVec = ColumnVector(4, intArrayOf(33, 19, 19, 11))
        assertEquals(colVec, mat1.rowVecNormSq())
    }

    @Test
    fun sum() {
        assertEquals(22.0, mat1.sum(), 0.00001)
    }

    @Test
    fun eltwiseMul() {
        val eltwiseMat = Matrix(4, 4, intArrayOf(
            16, 9, 4, 4,
            0, 1, 9, 9,
            0, 1, 9, 9,
            0, 9, 1, 1
        ))
        assertEquals(eltwiseMat, mat1.eltwiseMul(mat1))
    }

    @Test
    fun rowWiseMean() {
        val colVec = ColumnVector(4, doubleArrayOf(2.75, 0.25, 1.25, 1.25))
        assertEquals(colVec, mat1.rowWiseMean())
    }

    @Test
    fun columnWiseMean() {
        val rowVec = RowVector(4, doubleArrayOf(1.0, 1.5, 0.75, 2.25))
        assertEquals(rowVec, mat1.columnWiseMean())
    }

    @Test
    fun map() {
        assertEquals(twiceMat1, mat1.map{it * 2})
    }

    @Test
    fun reshape() {
        val reshapeMat1 = Matrix(8, 2, intArrayOf(
            4, 3, 2, 2,
            0, 1, -3, 3,
            0, -1, 3, 3,
            0, 3, 1, 1
        ))
        assertEquals(reshapeMat1, mat1.reshape(8, -1))
    }

    @Test
    fun toRowVector() {
        val mat = Matrix(1, 4, IntArray(4))
        val rowVec = RowVector(4, IntArray(4))
        assertEquals(rowVec, mat.toRowVector())
    }

    @Test
    fun toColVector() {
        val mat = Matrix(4, 1, IntArray(4))
        val colVec = ColumnVector(4, IntArray(4))
        assertEquals(colVec, mat)
    }

    @Test
    fun toComplex() {
        val complexMat1 = ComplexMatrix(4, 4, arrayOf(
            4.R, 3.R, 2.R, 2.R,
            0.R, 1.R, -3.R, 3.R,
            0.R, -1.R, 3.R, 3.R,
            0.R, 3.R, 1.R, 1.R
        ))
        assertEquals(complexMat1, mat1.toComplex())
    }

    @Test
    fun copy() {
        val newMat = mat1.copy()
        assertEquals(mat1, newMat)
        assertFalse(mat1 === newMat)
    }
}