import complex.ComplexMatrix
import complex.ComplexTensor
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Test
import utils.I
import utils.R
import utils.times

internal class ComplexTensorTest {
    private val tensor1 = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {(it+1).R})
    private val negTensor1 = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {-(it+1).R})
    private val twiceTensor1 = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {2 * (it+1).R})
    private val tensor2 = ComplexTensor(intArrayOf(4, 3), arrayOf(
        1.R, 2.R, 3.R,
        1.R, 2.R, 3.R,
        1.R, 2.R, 3.R,
        1.R, 2.R, 3.R
    ))

    @Test
    fun get() {
        assertEquals(18.0.R, tensor1[intArrayOf(0, 1, 1, 1)])
        val subTensor = ComplexTensor(intArrayOf(3, 4), Array(12) {(it+13).R})
        assertEquals(subTensor, tensor1[0][1])
    }

    @Test
    fun set() {
        val tensor = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {(it+1).R})
        tensor[intArrayOf(0, 1, 1, 1)] = 10.R + 27.I
        assertEquals(10.R + 27.I, tensor[intArrayOf(0, 1, 1, 1)])
    }

    @Test
    operator fun unaryPlus() {
        assertEquals(tensor1, +tensor1)
    }

    @Test
    operator fun unaryMinus() {
        assertEquals(negTensor1, -tensor1)
    }

    @Test
    fun plus() {
        assertEquals(twiceTensor1, tensor1 + tensor1)
    }

    @Test
    fun minus() {
        assertEquals(tensor1, twiceTensor1 - tensor1)
    }

    @Test
    fun times() {
        assertEquals(twiceTensor1, tensor1 * 2)
        assertEquals(twiceTensor1, 2 * tensor1)
        assertEquals(twiceTensor1, 2.R * tensor1)

        val tensorProduct = ComplexTensor(intArrayOf(1, 2, 3, 3), arrayOf(
            10.R, 20.R, 30.R, 26.R, 52.R, 78.R, 42.R, 84.R, 126.R,
            58.R, 116.R, 174.R, 74.R, 148.R, 222.R, 90.R, 180.R, 270.R))

        assertEquals(tensorProduct, tensor1 * tensor2)
    }

    @Test
    fun div() {
        assertEquals(tensor1, twiceTensor1 / 2)
        assertEquals(tensor1, twiceTensor1 / 2.R)
    }

    @Test
    fun plusAssign() {
        var tensor = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {(it+1).R})
        tensor += tensor1
        assertEquals(twiceTensor1, tensor)
    }

    @Test
    fun minusAssign() {
        var twiceTensor = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {2 * (it+1).R})
        twiceTensor -= tensor1
        assertEquals(tensor1, twiceTensor)
    }

    @Test
    fun timesAssign() {
        var tensor = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) {(it+1).R})
        tensor *= 2
        assertEquals(twiceTensor1, tensor)

    }

    @Test
    fun toComplexMatrix() {
        val mat = ComplexMatrix(4, 3, arrayOf(
            1.R, 2.R, 3.R,
            1.R, 2.R, 3.R,
            1.R, 2.R, 3.R,
            1.R, 2.R, 3.R
        ))
        assertEquals(mat, tensor2.toComplexMatrix())
    }

    @Test
    fun reshape() {
        val tensor1Reshape = ComplexTensor(intArrayOf(6, 4), Array(1*2*3*4) {(it+1).R})

        assertEquals(tensor1Reshape, tensor1.reshape(intArrayOf(6, -1)))
    }

    @Test
    fun flatten() {
        val tensor1Flatten = ComplexTensor(intArrayOf(24), Array(1*2*3*4) {(it+1).R})
        assertEquals(tensor1Flatten, tensor1.flatten())
    }

    @Test
    fun concat() {
        val concatTensor = ComplexTensor(intArrayOf(1, 2, 6, 4), arrayOf(
            1.R, 2.R, 3.R, 4.R, 5.R, 6.R, 7.R, 8.R, 9.R, 10.R, 11.R, 12.R,
            1.R, 2.R, 3.R, 4.R, 5.R, 6.R, 7.R, 8.R, 9.R, 10.R, 11.R, 12.R,
            13.R, 14.R, 15.R, 16.R, 17.R, 18.R, 19.R, 20.R, 21.R, 22.R, 23.R, 24.R,
            13.R, 14.R, 15.R, 16.R, 17.R, 18.R, 19.R, 20.R, 21.R, 22.R, 23.R, 24.R
        ))
        assertEquals(concatTensor, tensor1.concat(tensor1, 2))
    }

    @Test
    fun map() {
        assertEquals(twiceTensor1, tensor1.map {it * 2})
    }

    @Test
    fun copy() {
        val tensor1Copy = tensor1.copy()
        assertEquals(tensor1, tensor1Copy)
        assertFalse(tensor1 === tensor1Copy)
    }
}