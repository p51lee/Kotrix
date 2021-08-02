import complex.ComplexColumnVector
import complex.ComplexMatrix
import complex.ComplexRowVector
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import utils.I
import utils.R
import utils.times
import utils.plus

internal class ComplexColumnVectorTest {
    private val vec = ComplexColumnVector(4, arrayOf(1.I, 2.R, 3.I, 4.R))
    private val minusVec = ComplexColumnVector(4, arrayOf((-1).I, (-2).R, (-3).I, (-4).R))
    private val twiceVec = ComplexColumnVector(4, arrayOf(2.I, 4.R, 6.I, 8.R))
    val mat = ComplexMatrix(4, 4) { i, j -> if (i == j) i + 1.R else 0.R}

    @Test
    fun get() {
        assertEquals(1.I, vec[0])
    }

    @Test
    fun set() {
        val vec1 = ComplexColumnVector(4, arrayOf(1.I, 2.R, 3.I, 4.R))
        vec1[0] = 1.R
        assertEquals(1.R, vec1[0])
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
        val vec1 = ComplexColumnVector(4, arrayOf(1.I, 4.R, 9.I, 16.R))
        assertEquals(twiceVec, 2 * vec)
        assertEquals(twiceVec, vec * 2)
        assertEquals(vec1, mat * vec)
    }

    @Test
    fun div() {
        assertEquals(vec, twiceVec / 2)
        assertEquals(vec, twiceVec / 2.R)
    }

    @Test
    fun transpose() {
        val vec1 = ComplexRowVector(4, arrayOf(1.I, 2.R, 3.I, 4.R))
        assertEquals(vec1, vec.transpose())
    }

    @Test
    fun getSubvector() {
        val subVec = ComplexColumnVector(2, arrayOf(2.R, 3.I))
        assertEquals(subVec, vec.getSubvector(1, 3))
    }

    @Test
    fun setSubvector() {
        val subVec = ComplexColumnVector(2, arrayOf(2.R, 3.R))
        val vec1 = ComplexColumnVector(4, arrayOf(1.I, 2.R, 3.I, 4.R))
        val vec2 = ComplexColumnVector(4, arrayOf(1.I, 2.R, 3.R, 4.R))
        vec1.setSubvector(1, 3, subVec)
        assertEquals(vec2, vec1)
    }

    @Test
    fun eltwiseMul() {
        val vec1 = ComplexColumnVector(4, arrayOf((-1).R, 4.R, (-9).R, 16.R))
        assertEquals(vec1, vec.eltwiseMul(vec))
    }

    @Test
    fun dotProduct() {
        assertEquals(10.R, vec.dotProduct(vec))
    }

    @Test
    fun crossProduct() {
        val vec1 = ComplexColumnVector(3, arrayOf(1.R, 2.R, 3.R))
        assertEquals(ComplexColumnVector(3), vec1.crossProduct(vec1))
    }

    @Test
    fun replicate() {
        val mat1 = ComplexMatrix(4, 4, arrayOf(
            1.I, 1.I, 1.I, 1.I,
            2.R, 2.R, 2.R, 2.R,
            3.I, 3.I, 3.I, 3.I,
            4.R, 4.R, 4.R, 4.R,
        ))
        assertEquals(mat1, vec.replicate(4))
    }

    @Test
    fun map() {
        assertEquals(twiceVec, vec.map {it * 2})
    }

    @Test
    fun copy() {
        val vec1 = vec.copy()
        assertEquals(vec1, vec)
        assertFalse(vec1 === vec)
    }
}