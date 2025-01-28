#include <cmath>
#include "Eigen/Dense"
#include "radarModel.h"

BoschRadar::BoschRadar(const Floating snr50db, const Floating rcs50, const Floating r50, const Floating pf, const Floating scale, const Floating rcs_min) : snr50_(BoschRadar::dbToLinear(snr50db)), rcs50_(rcs50), r50_(r50), pf_(pf), scale_(scale), rcsMin_(rcs_min) {}

Floating BoschRadar::computeSnr(const Floating range, const Floating phi, const Floating rcs)
{
    return snr50_ * (rcs / rcs50_) * std::pow(r50_ / range, 4) * gaussianGain(phi);
}

ArrayXF BoschRadar::computeSnr(const Eigen::Ref<const ArrayXF> &range, const Eigen::Ref<const ArrayXF> &phi, const Floating rcs)
{
    return snr50_ * (rcs / rcs50_) * Eigen::pow(r50_ / range, 4) * gaussianGain(phi);
}

Floating BoschRadar::computePd(const Floating snr)
{
    return std::pow(pf_, 1.0 / (1.0 + snr));
}

ArrayXF BoschRadar::computePd(const Eigen::Ref<const ArrayXF> &snr)
{
    return Eigen::pow(pf_, 1.0 / (1.0 + snr));
}

Floating BoschRadar::gaussianGain(const Floating phi)
{
    const Floating exponent = phi / scale_;
    return std::exp(-0.5f * exponent * exponent);
}

ArrayXF BoschRadar::gaussianGain(const Eigen::Ref<const ArrayXF> &phi)
{
    return (-0.5 * (phi / scale_).square()).exp();
}

Floating BoschRadar::simRcs(const Floating r, const Floating rcs, const bool addNoise)
{
    constexpr Floating randMax = static_cast<Floating>(RAND_MAX);
    const Floating dropAt = rcs * 2.0;
    const Floating noise = addNoise ? r * ((static_cast<Floating>(std::rand()) / randMax) + 0.5) / 10.0 : 0.0;
    if (r > dropAt)
    {
        return rcs + noise;
    }
    else
    {
        const Floating slope = (rcs - rcsMin_) / dropAt;
        return slope * r + rcsMin_ + noise;
    }
}

ArrayXF BoschRadar::simRcs(const Eigen::Ref<const ArrayXF> &r, const Floating rcs, const bool addNoise)
{
    const Floating dropAt = rcs * 2.0;
    const Floating slope = (rcs - rcsMin_) / dropAt;
    ArrayXF rcsSim = slope * r + rcsMin_;
    rcsSim = (r > dropAt).select(rcs, rcsSim);
    if (addNoise)
    {
        rcsSim += r * ((ArrayXF::Random(rcsSim.rows(), rcsSim.cols()) / 2.0) + 1.0) / 10.0;
    }

    return rcsSim;
}
