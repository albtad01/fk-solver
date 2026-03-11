Mesh.ScalingFactor = 1;
Merge "brain-h3.0.stl";

// Define a surface loop and volume
Surface Loop(1) = {1};  // Gmsh usually assigns the STL surface to ID 1
Volume(1) = {1};

// Mesh.CharacteristicLengthMin = 0.2;
// Mesh.CharacteristicLengthMax = 0.2;

// Define physical groups (optional but useful for BCs later)
Physical Surface("boundary") = {1};
Physical Volume("volume") = {1};
