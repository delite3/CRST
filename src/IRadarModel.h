#pragma once
#include <cmath>
#include "Eigen/Dense"

using Floating = double;
using ArrayXF = Eigen::Array<Floating, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class RadarModel
{
public:
    virtual ~RadarModel() = default;
    virtual Floating computeSnr(const Floating range, const Floating phi, const Floating rcs) = 0;
    virtual ArrayXF computeSnr(const Eigen::Ref<const ArrayXF> &range, const Eigen::Ref<const ArrayXF> &phi, const Floating rcs) = 0;
    virtual Floating computePd(const Floating snr) = 0;
    virtual ArrayXF computePd(const Eigen::Ref<const ArrayXF> &snr) = 0;
    virtual Floating simRcs(const Floating r, const Floating rcs, const bool addNoise) = 0;
    virtual ArrayXF simRcs(const Eigen::Ref<const ArrayXF> &r, const Floating rcs, const bool addNoise) = 0;

    static Floating dbToLinear(const Floating db)
    {
        return std::pow(10, db / 10.0);
    };
};