from pathlib import Path
import tempfile
import unittest
from PIL import Image

from prepare_dataset import prepare, voc_labels


def annotation(width=100, height=80, xmax=30):
    return f"""<annotation><size><width>{width}</width><height>{height}</height></size>
    <object><name>acne</name><bndbox><xmin>11</xmin><ymin>21</ymin>
    <xmax>{xmax}</xmax><ymax>40</ymax></bndbox></object></annotation>"""


class DatasetTests(unittest.TestCase):
    def test_voc_coordinate_conversion_and_dimension_check(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            image, xml = folder / "face.png", folder / "face.xml"
            Image.new("RGB", (100, 80)).save(image)
            xml.write_text(annotation())
            row = voc_labels(xml, image).split()
            self.assertEqual(row[0], "0")
            self.assertEqual(list(map(float, row[1:])), [0.2, 0.375, 0.2, 0.25])
            xml.write_text(annotation(width=90))
            with self.assertRaises(ValueError):
                voc_labels(xml, image)

    def test_preparation_deduplicates_and_separates_splits(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            source = folder / "raw"
            source.mkdir()
            for i in range(20):
                stem = f"levle0_{i}"
                Image.new("RGB", (100, 80), (i*10, 40, 90)).save(source / f"{stem}.png")
                (source / f"{stem}.xml").write_text(annotation())
            (source / "duplicate.png").write_bytes((source / "levle0_0.png").read_bytes())
            (source / "duplicate.xml").write_text(annotation())
            manifest = prepare(source, folder / "prepared")
            sets = [{row["sha256"] for row in items} for items in manifest["splits"].values()]
            self.assertEqual(sum(map(len, sets)), 20)
            self.assertEqual(manifest["duplicates_removed"], 1)
            for i, hashes in enumerate(sets):
                for other in sets[i+1:]:
                    self.assertFalse(hashes & other)
            with self.assertRaises(ValueError):
                prepare(source, folder / "prepared")


if __name__ == "__main__":
    unittest.main()
