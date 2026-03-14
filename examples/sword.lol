union(
    translate(0.0, 1.2, 0.0,
        subtract(
            scale_non_uniform(0.12, 1.0, 0.6, diamond(1.0, 2.6)),
            translate(0.0, -0.3, 0.0, box3d(0.2, 1.8, 0.05))
        )
    ),
    translate(0.0, -1.3, 0.0,
        smooth_union(0.2,
            scale_non_uniform(0.3, 0.2, 1.6, diamond(1.0, 1.0)),
            octahedron(0.4)
        )
    ),
    translate(0.0, -2.5, 0.0,
        smooth_union(0.1,
            cylinder(0.12, 0.8),
            translate(0.0, -0.9, 0.0, diamond(0.3, 0.4))
        )
    )
)
