import operations.*
import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import real.Matrix

internal class OperationTest {
    val mat = Matrix(5, 5, intArrayOf(
        3, 0, 0, 3, 0,
        -3, 0, -2, 0, 0,
        0, -1, 0, 0, -3,
        0, 0, 0, 3, 3,
        0, -1, 2, 0, 1
    ))
    @Test
    fun determinant() {
        assertEquals(-18.0, mat.determinant(), 0.00001)
        assertEquals(-54.0, mat.cofactorMatrix(0, 1).determinant(), 0.00001)
    }

    @Test
    fun inverse() {
        val invMat = Matrix(5, 5, doubleArrayOf(
            4.0/3.0, 1.0, -1.0, -4.0/3.0, 1.0,
            -3.0, -3.0, 2.0, 3.0, -3.0,
            -2.0, -2.0, 3.0/2.0, 2.0, -3.0/2.0,
            -1.0, -1.0, 1.0, 4.0/3.0, -1.0,
            1.0, 1.0, -1.0, -1.0, 1.0
        ))
        assertEquals(invMat, mat.inverse())
    }

    @Test
    fun rowEchelonForm() {
        val mat1 = Matrix(4, 4, intArrayOf(
            1, 1, 1, 1,
            1, 1, 1, 1,
            0, 1, 2, 3,
            0, 1, 2, 3
        ))
        val reducedRowEchelon = Matrix(4, 4, intArrayOf(
            1, 0, -1, -2,
            0, 1, 2, 3,
            0, 0, 0, 0,
            0, 0, 0, 0
        ))
        assertEquals(reducedRowEchelon, mat1.rowEchelonForm())
    }

    @Test
    fun plu() {
        val plu = mat.plu()
        val p = plu[0]
        val l = plu[1]
        val u = plu[2]
        assertEquals(mat, p * l * u)
    }

    @Test
    fun qr() {
        val expectedQ = Matrix(5, 5, doubleArrayOf(
            0.7071067812,	0.0,	        -0.5,	0.2236067978,	0.4472135955,
            -0.7071067812,	0.0,	        -0.5,	0.2236067978,	0.4472135955,
            0.0,	        -0.7071067812,	-0.5,	-0.2236067978,	-0.4472135955,
            0.0,	        0.0,	        0.0,	0.894427191,	-0.4472135955,
            0.0,	        -0.7071067812,	0.5,	0.2236067978,	0.4472135955,
        ))
        val expectedR = Matrix(5, 5, doubleArrayOf(
            4.242640687,	0.0,	        1.414213562,	2.121320344,	0.0,
            0.0,	        1.414213562,    -1.414213562,	0.0,	        1.414213562,
            0.0,	        0.0,	        2.0,	        -1.5,	        2.0,
            0.0,	        0.0,	        0.0,	        3.354101966,	3.577708764,
            0.0,	        0.0,	        0.0,	        0.0,	        0.4472135955
        ))

        val qr = mat.qr()
        val actualQ = qr[0]
        val actualR = qr[1]
        assertTrue(actualQ.pseudoEquals(expectedQ))
        assertTrue(actualR.pseudoEquals(expectedR))
    }

    @Test
    fun eig() {
        val mat1 = Matrix(3, 3, intArrayOf(
            8, 1, 6, 3, 5, 7, 4, 9, 2
        ))
        val eig = mat1.eig()
        val eigenVectorMat = eig[0]
        val eigenValueMat = eig[1]
        val product = mat1.toComplex() * eigenVectorMat
        val eigProduct = eigenVectorMat * eigenValueMat
        assertTrue(product.pseudoEquals(eigProduct))
    }

    @Test
    fun svd() {
        val svdMat = Matrix(2, 3, intArrayOf(
            3, 2, 2,
            2, 3, -2
        ))
        val svd = svdMat.svd()
        assertTrue(svdMat.toComplex().pseudoEquals(svd[0] * svd[1] * svd[2]))
    }
}