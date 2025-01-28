#pragma once
#include "IRadarModel.h"

class BoschRadar : public RadarModel
{
public:
    BoschRadar(const Floating snr50db, const Floating rcs50, const Floating r50, const Floating pf, const Floating scale, const Floating rcs_min);
    Floating computeSnr(const Floating range, const Floating phi, const Floating rcs);
    ArrayXF computeSnr(const Eigen::Ref<const ArrayXF> &range, const Eigen::Ref<const ArrayXF> &phi, const Floating rcs);
    Floating computePd(const Floating snr);
    ArrayXF computePd(const Eigen::Ref<const ArrayXF> &snr);
    Floating gaussianGain(const Floating phi);
    ArrayXF gaussianGain(const Eigen::Ref<const ArrayXF> &phi);
    Floating simRcs(const Floating r, const Floating rcs, const bool addNoise);
    ArrayXF simRcs(const Eigen::Ref<const ArrayXF> &r, const Floating rcs, const bool addNoise);

private:
    const Floating snr50_;
    const Floating rcs50_;
    const Floating r50_;
    const Floating pf_;
    const Floating scale_;
    const Floating rcsMin_;
};