package complex

import utils.I
import utils.R
import utils.toFormattedString
import kotlin.math.*

/**
 * Represents complex number using two [Double] values.
 *
 * @property re real part.
 * @property im imaginary part.
 * @constructor Converts real part and imaginary part from [Number] to [Double].
 *
 * @param real real part.
 * @param imaginary imaginary part.
 */
class ComplexDouble (real: Number, imaginary: Number) {
    var re = real.toDouble()
    var im = imaginary.toDouble()

    operator fun unaryPlus() = this

    operator fun unaryMinus() = ComplexDouble(-re, -im)

    operator fun plus(other: ComplexDouble) = ComplexDouble(re + other.re, im + other.im)

    operator fun plus(other: Number) = ComplexDouble(re + other.toDouble(), im)

    operator fun minus(other: ComplexDouble) = ComplexDouble(re - other.re, im - other.im)

    operator fun minus(other: Number) = ComplexDouble(re - other.toDouble(), im)

    operator fun times(other: ComplexDouble) = ComplexDouble(re*other.re - im*other.im, re*other.im + im*other.re)

    operator fun times(other: Number) = ComplexDouble(re*other.toDouble(), im*other.toDouble())

    operator fun times(other: ComplexTensor) = other * this

    operator fun times(other: ComplexMatrix) = other * this

    operator fun times(other: ComplexRowVector) = other * this

    operator fun times(other: ComplexColumnVector) = other * this

    operator fun div(other: ComplexDouble) = this * other.conj() / (other.re.pow(2) + other.im.pow(2))

    operator fun div(other: Number) = ComplexDouble(re/other.toDouble(), im/other.toDouble())

    override operator fun equals(other: Any?): Boolean {
        return if (other is ComplexDouble) {
            re == other.re && im == other.im
        } else false
    }

    /**
     * Complex conjugate.
     *
     * @return A new class instance of the complex conjugate.
     */
    fun conj() = ComplexDouble(re, -im)

    /**
     * Absolute value of a complex number.
     *
     * @return absolute value.
     */
    fun abs() = sqrt(re.pow(2) + im.pow(2))

    /**
     * Argument of a complex number from -PI to PI radians.
     *
     * @return argument.
     */
    fun arg() = atan2(im, re)

    /**
     * Multiply to the power of n.
     *
     * @param n an integer.
     * @return complex number to the n-th power.
     */
    fun pow(n: Int) = polarForm(abs().pow(n), arg() * n)

    /**
     * Multiply to the power of k.
     *
     * @param k a real number.
     * @return complex number to the k-th power.
     */
    fun pow(k: Double) = polarForm(abs().pow(k), arg() * k)

    /**
     * Most Generalized form of [pow].
     *
     * @param z a complex number.
     * @return complex number power [z].
     */
    fun pow(z: ComplexDouble): ComplexDouble {
        val ln = ln(this.abs()).R + this.arg().I
        val exponent = ln * z
        return polarForm(exp(exponent.re), exponent.im)
    }

    /**
     * Stringification. Uniform size in range (-10000, +10000)
     *
     * @return string in cartesian form.
     */
    override fun toString(): String {
        return if (im >= 0) " ${re.toFormattedString()}+${im.toFormattedString()}i "
        else " ${re.toFormattedString()}-${(-im).toFormattedString()}i "
    }

    companion object {
        /**
         * Generates a complex number in polar form.
         *
         * @param r absolute value.
         * @param theta argument in radian.
         */
        fun polarForm(r: Double, theta: Double) = ComplexDouble(r* cos(theta), r* sin(theta))
    }
}
