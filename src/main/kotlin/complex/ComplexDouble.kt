package complex

import utils.toFormattedString
import kotlin.math.*

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

    fun conj() = ComplexDouble(re, -im)

    fun abs() = sqrt(re.pow(2) + im.pow(2))

    fun arg() = atan2(im, re)

    fun pow(n: Int) = polarForm(abs().pow(n), arg() * n)

    fun pow(k: Double) = polarForm(abs().pow(k), arg() * k)

    override fun toString(): String {
        return if (im >= 0) " ${re.toFormattedString()}+${im.toFormattedString()}i "
        else " ${re.toFormattedString()}-${(-im).toFormattedString()}i "
    }

    companion object {
        fun polarForm(r: Double, theta: Double) = ComplexDouble(r* cos(theta), r* sin(theta))
    }
}
