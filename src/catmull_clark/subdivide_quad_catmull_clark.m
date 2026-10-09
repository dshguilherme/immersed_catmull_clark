function [V_out, F_out] = subdivide_quad_catmull_clark(V, F)
% SUBDIVIDE_QUAD_CATMULL_CLARK Performs one Catmull-Clark subdivision step on quad mesh
%
% Inputs:
%   V - N x 2 or N x 3 coordinates of vertices
%   F - M x 4 quad connectivity (1-based indices)
%
% Outputs:
%   V_out - (N + M + nEdges) x D new vertex coordinates
%   F_out - (4*M) x 4 refined quad connectivity

dim = size(V, 2);
numV = size(V, 1);
numF = size(F, 1);

% 1. Face Points: centroid of each face
facePoints = zeros(numF, dim);
for f = 1:numF
    facePoints(f, :) = mean(V(F(f, :), :), 1);
end

% 2. Extract unique edges and build face-edge adjacency
% Each quad has 4 edges: (1-2), (2-3), (3-4), (4-1)
rawEdges = [F(:, [1, 2]); F(:, [2, 3]); F(:, [3, 4]); F(:, [4, 1])];
faceOfEdge = repmat((1:numF)', 4, 1);
sortedEdges = sort(rawEdges, 2);

[uniqueEdges, ~, edgeMap] = unique(sortedEdges, 'rows');
numEdges = size(uniqueEdges, 1);

% Map each edge to adjacent faces
edgeToFaces = cell(numEdges, 1);
for k = 1:numel(edgeMap)
    eIdx = edgeMap(k);
    fIdx = faceOfEdge(k);
    edgeToFaces{eIdx} = [edgeToFaces{eIdx}, fIdx];
end

% 3. Edge Points
% Interior edge: average of the two endpoints and two adjacent face points
% Boundary edge: midpoint of the two endpoints
edgePoints = zeros(numEdges, dim);
isBoundaryEdge = false(numEdges, 1);
for e = 1:numEdges
    v1 = uniqueEdges(e, 1);
    v2 = uniqueEdges(e, 2);
    adjF = edgeToFaces{e};
    if numel(adjF) == 2
        edgePoints(e, :) = 0.25 * (V(v1, :) + V(v2, :) + facePoints(adjF(1), :) + facePoints(adjF(2), :));
    else
        isBoundaryEdge(e) = true;
        edgePoints(e, :) = 0.5 * (V(v1, :) + V(v2, :));
    end
end

% 4. Vertex Points (updated existing vertices)
vertexToFaces = cell(numV, 1);
vertexToEdges = cell(numV, 1);
for f = 1:numF
    for vi = F(f, :)
        vertexToFaces{vi} = [vertexToFaces{vi}, f];
    end
end
for e = 1:numEdges
    v1 = uniqueEdges(e, 1);
    v2 = uniqueEdges(e, 2);
    vertexToEdges{v1} = [vertexToEdges{v1}, e];
    vertexToEdges{v2} = [vertexToEdges{v2}, e];
end

newV = zeros(numV, dim);
for v = 1:numV
    adjE = vertexToEdges{v};
    bEdges = adjE(isBoundaryEdge(adjE));
    
    if ~isempty(bEdges)
        % Boundary vertex rule
        % New position = (3/4)*P + (1/8)*(R1 + R2) where R1, R2 are boundary neighbor vertices
        bNeighbors = [];
        for be = bEdges
            ev = uniqueEdges(be, :);
            otherV = ev(ev ~= v);
            bNeighbors = [bNeighbors, otherV];
        end
        if numel(bNeighbors) >= 2
            newV(v, :) = 0.75 * V(v, :) + 0.125 * (V(bNeighbors(1), :) + V(bNeighbors(2), :));
        else
            newV(v, :) = V(v, :);
        end
    else
        % Interior vertex rule: P_new = (F_avg + 2*R_avg + (n-3)*P) / n
        n = numel(vertexToFaces{v});
        fAvg = mean(facePoints(vertexToFaces{v}, :), 1);
        % Midpoints of adjacent edges
        edgeMidAvg = zeros(1, dim);
        for ke = 1:numel(adjE)
            ev = uniqueEdges(adjE(ke), :);
            edgeMidAvg = edgeMidAvg + 0.5 * (V(ev(1), :) + V(ev(2), :));
        end
        edgeMidAvg = edgeMidAvg / numel(adjE);
        newV(v, :) = (fAvg + 2 * edgeMidAvg + (n - 3) * V(v, :)) / n;
    end
end

% Combine all new vertices into global vertex table
% Index layout:
%   1 : numV                       -> updated original vertices
%   numV + (1 : numF)              -> face points
%   numV + numF + (1 : numEdges)   -> edge points
V_out = [newV; facePoints; edgePoints];

facePtOffset = numV;
edgePtOffset = numV + numF;

% 5. Form 4 new quads per original quad
% Each original face F = [v1, v2, v3, v4] with edges e1(1-2), e2(2-3), e3(3-4), e4(4-1)
% and face centroid cf produces 4 sub-quads:
%   Quad 1: [v1, e1, cf, e4]
%   Quad 2: [v2, e2, cf, e1]
%   Quad 3: [v3, e3, cf, e2]
%   Quad 4: [v4, e4, cf, e3]
F_out = zeros(4 * numF, 4);
for f = 1:numF
    v1 = F(f, 1); v2 = F(f, 2); v3 = F(f, 3); v4 = F(f, 4);
    cf = facePtOffset + f;
    
    e1 = edgePtOffset + edgeMap(f);                % edge 1-2
    e2 = edgePtOffset + edgeMap(numF + f);         % edge 2-3
    e3 = edgePtOffset + edgeMap(2*numF + f);       % edge 3-4
    e4 = edgePtOffset + edgeMap(3*numF + f);       % edge 4-1
    
    qIdx = (f - 1) * 4;
    F_out(qIdx + 1, :) = [v1, e1, cf, e4];
    F_out(qIdx + 2, :) = [v2, e2, cf, e1];
    F_out(qIdx + 3, :) = [v3, e3, cf, e2];
    F_out(qIdx + 4, :) = [v4, e4, cf, e3];
end

end
