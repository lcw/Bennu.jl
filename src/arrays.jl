Tullio.storage_type(s::StructArray) =
    promote_type(map(Tullio.storage_type, StructArrays.components(s))...)
