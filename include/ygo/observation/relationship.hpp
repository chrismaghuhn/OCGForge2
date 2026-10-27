#pragma once

#include <string>

#include "ygo/observation/observed_zone.hpp"

namespace ygo::observation {

enum class RelationshipKind {
    XyzMaterial,
    Equip,
    Target,
};

std::string relationship_kind_name(RelationshipKind kind);

struct Relationship {
    RelationshipKind kind = RelationshipKind::Target;
    ObservationLocator source;
    ObservationLocator target;
};

}  // namespace ygo::observation
