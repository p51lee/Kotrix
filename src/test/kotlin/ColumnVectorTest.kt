import complex.ComplexColumnVector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import real.ColumnVector
import real.Matrix
import real.RowVector
import utils.R
import utils.times

internal class ColumnVectorTest {
    private val vec = ColumnVector(4, intArrayOf(1, 2, 3, 4))
    private val minusVec = ColumnVector(4, intArrayOf(-1, -2, -3, -4))
    private val twiceVec = ColumnVector(4, intArrayOf(2, 4, 6, 8))
    val mat = Matrix(4, 4) { i, j -> if (i == j) i + 1 else 0}

    @Test
    fun get() {
        assertEquals(3.0, vec[2], 0.00001)
    }

    @Test
    fun set() {
        val vec1 = ColumnVector(4, intArrayOf(1, 2, 3, 4))
        vec1[2] = 1102
        assertEquals(1102.0, vec1[2], 0.00001)
    }

    @Test
    operator fun unaryPlus() {
        assertEquals(vec, +vec)
    }

    @Test
    operator fun unaryMinus() {
        assertEquals(minusVec, -vec)
    }

    @Test
    fun plus() {
        assertEquals(twiceVec, vec + vec)
    }

    @Test
    fun minus() {
        assertEquals(vec, twiceVec - vec)
    }

    @Test
    fun times() {
        assertEquals(twiceVec, 2 * vec)
        assertEquals(twiceVec, vec * 2)
        assertEquals(
            ColumnVector(4, intArrayOf(1, 4, 9, 16)),
            mat * vec
        )
    }

    @Test
    fun div() {
        assertEquals(vec, twiceVec / 2)
    }

    @Test
    fun transpose() {
        val colVec = RowVector(4, intArrayOf(1, 2, 3, 4))
        assertEquals(colVec, vec.transpose())
    }

    @Test
    fun getSubvector() {
        val subVec = ColumnVector(2, intArrayOf(2, 3))
        assertEquals(subVec, vec.getSubvector(1, 3))
    }

    @Test
    fun setSubvector() {
        val vec1 = ColumnVector(4, intArrayOf(1, 2, 3, 4))
        val subVec = ColumnVector(2, intArrayOf(10, 27))
        val resultVec = ColumnVector(4, intArrayOf(1, 10, 27, 4))
        vec1.setSubvector(1, 3, subVec)
        assertEquals(resultVec, vec1)
    }

    @Test
    fun eltwiseMul() {
        val mulVec = ColumnVector(4, intArrayOf(1, 4, 9, 16))
        assertEquals(mulVec, vec.eltwiseMul(vec))
    }

    @Test
    fun dotProduct() {
        val colVec = ColumnVector(4, intArrayOf(1, 2, 3, 4))
        assertEquals(30.0, vec.dotProduct(colVec), 0.00001)
        assertEquals(30.0, vec.dotProduct(vec), 0.00001)
    }

    @Test
    fun crossProduct() {
        val vec3 = ColumnVector(3, intArrayOf(1, 2, 3))
        assertEquals(ColumnVector(3), vec3.crossProduct(vec3))
    }

    @Test
    fun replicate() {
        val repMat = Matrix(4, 4, intArrayOf(1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4))
        assertEquals(repMat, vec.replicate(4))
    }

    @Test
    fun map() {
        assertEquals(twiceVec, vec.map {2*it})
    }

    @Test
    fun toComplex() {
        val compVec = ComplexColumnVector(4, arrayOf(1.R, 2.R, 3.R, 4.R))
        assertEquals(compVec, vec.toComplex())
    }

    @Test
    fun copy() {
        val copyVec = vec.copy()
        assertEquals(copyVec, vec)
        assertFalse(copyVec === vec)
    }

    @Test
    fun proj() {
        val vec1 = ColumnVector(3, intArrayOf(1, 0, 3))
        val vec2 = ColumnVector(3, intArrayOf(-1, 4, 2))
        val projVec = ColumnVector(3, doubleArrayOf(0.5, 0.0, 1.5))
        assertEquals(projVec, vec2.proj(vec1))
    }

    @Test
    fun normalize() {
        val vec1 = ColumnVector(2, intArrayOf(3, 4))
        val normVec = ColumnVector(2, doubleArrayOf(3.0 / 5.0, 4.0 / 5.0))
        assertEquals(normVec, vec1.normalize())
    }
}