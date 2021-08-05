import complex.ComplexColumnVector
import complex.ComplexMatrix
import complex.ComplexRowVector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import utils.I
import utils.R
import utils.times

internal class ComplexMatrixTest {
    private val mat = ComplexMatrix(3, 3, arrayOf(
        7.R,        0.R,        1.R + 1.I,
        0.R,        1.R,        9.I,
        1.R - 1.I,  (-4).I,     (-10).R
    ))

    private val negativeMat = ComplexMatrix(3, 3, arrayOf(
        (-7).R, 0.R, (-1).R + (-1).I,
        0.R, (-1).R, (-9).I,
        (-1).R + 1.I, 4.I, 10.R
    ))

    val twiceMat = ComplexMatrix(3, 3, arrayOf(
        14.R, 0.R, 2.R + 2.I,
        0.R, 2.R, 18.I,
        2.R - 2.I, (-8).I, (-20).R
    ))

    @Test
    fun get() {
        assertEquals(1.R, mat[1, 1])
    }

    @Test
    fun set() {
        val mat1 = ComplexMatrix(3, 3, arrayOf(
            7.R,        0.R,    1.R + 1.I,
            0.R,        1.R,    9.I,
            1.R - 1.I,  (-4).I, (-10).R
        ))
        mat1[1, 1] = 2018.R + 1102.I
        assertEquals(2018.R + 1102.I, mat1[1, 1])
    }

    @Test
    operator fun unaryPlus() {
        assertEquals(mat, +mat)
    }

    @Test
    operator fun unaryMinus() {
        assertEquals(negativeMat, -mat)
    }

    @Test
    fun plus() {
        assertEquals(twiceMat, mat + mat)
    }

    @Test
    fun minus() {
        assertEquals(mat, twiceMat - mat)
    }

    @Test
    fun times() {
        val mat1 = ComplexMatrix(3, 3, arrayOf(
            51.R,           4.R + (-4).I,   (-3).R + (-3).I,
            9.R + 9.I,      37.R,           (-81).I,
            (-3).R + 3.I,   36.I,           138.R
        ))
        assertEquals(twiceMat, 2 * mat)
        assertEquals(twiceMat, mat * 2)
        assertEquals(twiceMat, 2.R * mat)
        assertEquals(twiceMat, mat * 2.R)
        assertEquals(mat1, mat * mat)
    }

    @Test
    fun div() {
        assertEquals(mat, twiceMat / 2)
        assertEquals(mat, twiceMat / 2.R)
    }

    @Test
    fun transpose() {
        val tMat = ComplexMatrix(3, 3, arrayOf(
            7.R,        0.R,        1.R - 1.I,
            0.R,        1.R,        (-4).I,
            1.R + 1.I,  9.I,     (-10).R
        ))
        assertEquals(tMat, mat.transpose())
    }

    @Test
    fun conjTrans() {
        val conjTransMat = ComplexMatrix(3, 3, arrayOf(
            7.R,        0.R,        1.R + 1.I,
            0.R,        1.R,        (4).I,
            1.R - 1.I,  (-9).I,     (-10).R
        ))
        assertEquals(conjTransMat, mat.conjTrans())
    }

    @Test
    fun getSubmatrix() {
        val subMatrix = ComplexMatrix(2, 2, arrayOf(
            7.R,    0.R,
            0.R,    1.R
        ))
        assertEquals(subMatrix, mat.getSubmatrix(0, 2, 0, 2))
    }

    @Test
    fun setSubmatrix() {
        val newSubMat = ComplexMatrix(2, 2, arrayOf(
            1.R, 0.R,
            2.R, 7.R
        ))
        val mat1 = ComplexMatrix(3, 3, arrayOf(
            7.R, 0.R, 1.R + 1.I,
            0.R, 1.R, 9.I,
            1.R - 1.I, (-4).I, (-10).R
        ))
        val newMat = ComplexMatrix(3, 3, arrayOf(
            1.R, 0.R, 1.R + 1.I,
            2.R, 7.R, 9.I,
            1.R - 1.I, (-4).I, (-10).R
        ))
        mat1.setSubmatrix(0, 2, 0, 2, newSubMat)
        assertEquals(newMat, mat1)

    }

    @Test
    fun cofactorMatrix() {
        val cofactorMatrix = ComplexMatrix(2, 2, arrayOf(
            7.R,    0.R,
            0.R,    1.R
        ))
        assertEquals(cofactorMatrix, mat.minorMatrix(2, 2))
    }

    @Test
    fun switchRow() {
        val mat1 = ComplexMatrix(3, 3, arrayOf(
            0.R, 1.R, 9.I,
            7.R, 0.R, 1.R + 1.I,
            1.R - 1.I, (-4).I, (-10).R
        ))
        assertEquals(mat1, mat.switchRow(0, 1))
    }

    @Test
    fun addRow() {
        val mat1 = ComplexMatrix(3, 3, arrayOf(
            7.R, 2.R, 1.R + 19.I,
            0.R, 1.R, 9.I,
            1.R - 1.I, (-4).I, (-10).R
        ))
        assertEquals(mat1, mat.addRow(1, 0, 2.0))
    }

    @Test
    fun concat() {
        val mat1 = ComplexMatrix(6, 3, arrayOf(
            7.R, 0.R, 1.R + 1.I,
            0.R, 1.R, 9.I,
            1.R - 1.I, (-4).I, (-10).R,
            7.R, 0.R, 1.R + 1.I,
            0.R, 1.R, 9.I,
            1.R - 1.I, (-4).I, (-10).R
        ))
        assertEquals(mat1, mat.concat(mat, 0))
    }

    @Test
    fun sum() {
        assertEquals(5.I, mat.sum())
    }

    @Test
    fun eltwiseMul() {
        val eltMat = ComplexMatrix(3, 3, arrayOf(
            49.R,       0.R,        2.I,
            0.R,        1.R,        (-81).R,
            (-2).I,     (-16).R,    100.R
        ))
        assertEquals(eltMat, mat.eltwiseMul(mat))
    }

    @Test
    fun map() {
        assertEquals(twiceMat, mat.map {2 * it})
    }

    @Test
    fun reshape() {
        val reshapeMat = ComplexMatrix(9, 1, arrayOf(
            7.R,        0.R,        1.R + 1.I,
            0.R,        1.R,        9.I,
            1.R - 1.I,  (-4).I,     (-10).R
        ))
        assertEquals(reshapeMat, mat.reshape(9, -1))
    }

    @Test
    fun toComplexRowVector() {
        val mat1 = ComplexMatrix(1, 9, arrayOf(
            7.R,        0.R,        1.R + 1.I,
            0.R,        1.R,        9.I,
            1.R - 1.I,  (-4).I,     (-10).R
        ))
        val rowVec = ComplexRowVector(9, arrayOf(
            7.R,        0.R,        1.R + 1.I,
            0.R,        1.R,        9.I,
            1.R - 1.I,  (-4).I,     (-10).R
        ))
        assertEquals(rowVec, mat1.toComplexRowVector())
    }

    @Test
    fun toComplexColVector() {
        val mat1 = ComplexMatrix(9, 1, arrayOf(
            7.R,        0.R,        1.R + 1.I,
            0.R,        1.R,        9.I,
            1.R - 1.I,  (-4).I,     (-10).R
        ))
        val colVec = ComplexColumnVector(9, arrayOf(
            7.R,        0.R,        1.R + 1.I,
            0.R,        1.R,        9.I,
            1.R - 1.I,  (-4).I,     (-10).R
        ))
        assertEquals(colVec, mat1.toComplexColVector())
    }

    @Test
    fun copy() {
        val newMat = mat.copy()
        assertEquals(mat, newMat)
        assertFalse(mat === newMat)
    }
}