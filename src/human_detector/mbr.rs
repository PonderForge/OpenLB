use geo::{coord, line_string, Area, BooleanOps, Coord, LineString, Polygon};

/// Minimum Bounding Rectangle.
#[derive(Clone, PartialEq)]
pub struct Mbr {
    ls: LineString,
    id: isize,
    confidence: f32,
    name: Option<String>,
}

impl Default for Mbr {
    fn default() -> Self {
        Self {
            ls: line_string![],
            id: -1,
            confidence: 0.,
            name: None,
        }
    }
}

impl std::fmt::Debug for Mbr {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Mbr")
            .field("vertices", &self.ls)
            .field("id", &self.id)
            .field("name", &self.name)
            .field("confidence", &self.confidence)
            .finish()
    }
}

impl Mbr {

    /// Computes the intersection over union (IoU) between this bounding box and another.
    pub fn iou(&self, other: &Self) -> f32 {
        self.intersect(other) / self.union(other)
    }

    /// Build from (cx, cy, width, height, radians)
    pub fn from_cxcywhr(cx: f64, cy: f64, w: f64, h: f64, r: f64) -> Self {
        // [[cos -sin], [sin cos]]
        let m = [
            [r.cos() * 0.5 * w, -r.sin() * 0.5 * h],
            [r.sin() * 0.5 * w, r.cos() * 0.5 * h],
        ];
        let c = coord! {
            x: cx,
            y: cy,
        };

        let a_ = coord! {
            x: m[0][0] + m[0][1],
            y: m[1][0] + m[1][1],
        };

        let b_ = coord! {
            x: m[0][0] - m[0][1],
            y: m[1][0] - m[1][1],
        };

        let v1 = c + a_;
        let v2 = c + b_;
        let v3 = c * 2. - v1;
        let v4 = c * 2. - v2;

        Self {
            ls: vec![v1, v2, v3, v4].into(),
            ..Default::default()
        }
    }

    pub fn intersect(&self, other: &Mbr) -> f32 {
        let p1 = Polygon::new(self.ls.clone(), vec![]);
        let p2 = Polygon::new(other.ls.clone(), vec![]);
        p1.intersection(&p2).unsigned_area() as f32
    }

    pub fn union(&self, other: &Mbr) -> f32 {
        let p1 = Polygon::new(self.ls.clone(), vec![]);
        let p2 = Polygon::new(other.ls.clone(), vec![]);
        p1.union(&p2).unsigned_area() as f32
    }
}