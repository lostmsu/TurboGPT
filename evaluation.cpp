#include "evaluation.h"
#include <cmath>
#include <stdexcept>

Evaluation evaluate(Engine *e, const TrainingConfig &o, int64_t size, int batches) {
    Evaluation r;
    r.positions.resize(o.model.context);
    std::vector<int64_t> endpoints(o.model.batch);
    std::vector<float> loss(o.model.context);
    for (int batch = 0; batch < batches; ++batch) {
        for (int b = 0; b < o.model.batch; ++b)
            endpoints[b] = 8 + hash_u64(0x12345678ULL + uint64_t(batch) * o.model.batch + b) %
                                   uint64_t(size - 8);
        check_status(tg_evaluate(e, endpoints.data(), loss.data()));
        for (int p = 0; p < o.model.context; ++p) {
            if (!std::isfinite(loss[p]))
                throw std::runtime_error("Non-finite evaluation loss");
            r.positions[p] += loss[p] / std::log(2.0) / batches;
        }
        r.batch_last.push_back(loss.back() / std::log(2.0));
    }
    r.last = r.positions.back();
    for (double b : r.batch_last)
        r.se += (b - r.last) * (b - r.last);
    r.se = std::sqrt(r.se / (batches - 1) / batches);
    // Requested diagnostic. Causality is separately verified by tests/verify.py.
    if (!(r.positions[1] * 10 > r.last))
        throw std::runtime_error("Position-loss diagnostic failed: 10 * pos1 <= final");
    return r;
}
