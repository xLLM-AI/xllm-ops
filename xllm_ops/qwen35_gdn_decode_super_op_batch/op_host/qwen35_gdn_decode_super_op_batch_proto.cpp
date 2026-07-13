#include "register/op_impl_registry.h"
#include "tiling_base/error_log.h"

namespace ops {

static ge::graphStatus InferShapeQwen35GdnDecodeSuperOpBatch(gert::InferShapeContext* context)
{
    const gert::Shape* qkvShape = context->GetInputShape(0);
    const gert::Shape* zShape = context->GetInputShape(1);
    const gert::Shape* convStateShape = context->GetInputShape(5);
    const gert::Shape* ssmStateShape = context->GetInputShape(8);
    OP_CHECK_NULL_WITH_CONTEXT(context, qkvShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, zShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, convStateShape);
    OP_CHECK_NULL_WITH_CONTEXT(context, ssmStateShape);

    *context->GetOutputShape(0) = *qkvShape;
    *context->GetOutputShape(1) = *convStateShape;
    *context->GetOutputShape(2) = *ssmStateShape;
    *context->GetOutputShape(3) = *zShape;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_INFERSHAPE(Qwen35GdnDecodeSuperOpBatch).InferShape(InferShapeQwen35GdnDecodeSuperOpBatch);

} // namespace ops
