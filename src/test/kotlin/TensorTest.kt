import complex.ComplexTensor
import org.junit.Assert.*
import org.junit.Test
import real.Matrix
import real.Tensor
import utils.R
import utils.times

internal class TensorTest {
    private val tensor1 = Tensor(intArrayOf(1,2,3,4), intArrayOf(
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
        13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
    ))

    private val negTensor1 = Tensor(intArrayOf(1,2,3,4), intArrayOf(
        -1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12,
        -13, -14, -15, -16, -17, -18, -19, -20, -21, -22, -23, -24
    ))

    private val twiceTensor1 = Tensor(intArrayOf(1,2,3,4), intArrayOf(
        2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
        26, 28, 30, 32, 34 ,36 ,38 ,40, 42, 44, 46, 48
    ))

    private val tensor2 = Tensor(intArrayOf(4, 3), intArrayOf(
        1, 2, 3,
        1, 2, 3,
        1, 2, 3,
        1, 2, 3
    ))

    @Test
    fun get() {
        assertEquals(18.0, tensor1[intArrayOf(0, 1, 1, 1)], 0.00001)
        val subTensor = Tensor(intArrayOf(3, 4), intArrayOf(
            13, 14, 15, 16,
            17, 18, 19, 20,
            21, 22, 23, 24
        ))
        assertEquals(subTensor, tensor1[0][1])
    }

    @Test
    fun set() {
        val tensor = Tensor(intArrayOf(1,2,3,4), intArrayOf(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ))
        tensor[intArrayOf(0, 1, 1, 1)] = 1102.0
        assertEquals(1102.0, tensor[intArrayOf(0, 1, 1, 1)], 0.00001)
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

        val tensorProduct = Tensor(intArrayOf(1, 2, 3, 3), intArrayOf(
            10, 20, 30, 26, 52, 78, 42, 84, 126,
            58, 116, 174, 74, 148, 222, 90, 180, 270
        ))

        assertEquals(tensorProduct, tensor1 * tensor2)
    }

    @Test
    fun div() {
        assertEquals(tensor1, twiceTensor1 / 2)
    }

    @Test
    fun timesAssign() {
        var tensor = Tensor(intArrayOf(1,2,3,4), intArrayOf(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ))
        tensor *= 2
        assertEquals(twiceTensor1, tensor)
    }

    @Test
    fun divAssign() {
        var twiceTensor = Tensor(intArrayOf(1,2,3,4), intArrayOf(
            2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24,
            26, 28, 30, 32, 34 ,36 ,38 ,40, 42, 44, 46, 48
        ))
        twiceTensor /= 2
        assertEquals(tensor1, twiceTensor)
    }

    @Test
    fun toMatrix() {
        val tensor2Mat = tensor2.toMatrix()
        assertTrue(tensor2Mat is Matrix)
        assertEquals(tensor2, tensor2Mat)
    }

    @Test
    fun reshape() {
        val tensor1Reshape = Tensor(intArrayOf(6, 4), intArrayOf(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ))
        assertEquals(tensor1Reshape, tensor1.reshape(intArrayOf(6, -1)))
    }

    @Test
    fun flatten() {
        val tensor1Flatten = Tensor(intArrayOf(24), intArrayOf(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ))
        assertEquals(tensor1Flatten, tensor1.flatten())
    }

    @Test
    fun concat() {
        val concatTensor = Tensor(intArrayOf(1, 2, 6, 4), intArrayOf(
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24,
            13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ))
        assertEquals(concatTensor, tensor1.concat(tensor1, 2))
    }

    @Test
    fun map() {
        assertEquals(twiceTensor1, tensor1.map {it * 2})
    }

    @Test
    fun toComplex() {
        val complexTensor = ComplexTensor(intArrayOf(1, 2, 3, 4), Array(1*2*3*4) { (it + 1).R })
        assertEquals(complexTensor, tensor1.toComplex())
    }

    @Test
    fun copy() {
        val tensor1Copy = tensor1.copy()
        assertEquals(tensor1, tensor1Copy)
        assertFalse(tensor1 === tensor1Copy)
    }
}