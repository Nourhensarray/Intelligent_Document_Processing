from collections import defaultdict


class DocumentLayoutBuilder:

    def __init__(self, line_tolerance=15):
        self.line_tolerance = line_tolerance

    def build(self, ocr_data):

        if not ocr_data:
            return []

        # Trier de haut vers bas
        items = sorted(
            ocr_data,
            key=lambda x: self.center_y(x["box"])
        )

        rows = defaultdict(list)

        for item in items:

            y = self.center_y(item["box"])

            found = False

            for row_y in list(rows.keys()):

                if abs(y-row_y) < self.line_tolerance:

                    rows[row_y].append(item)

                    found = True

                    break

            if not found:

                rows[y].append(item)

        lines = []

        for row in sorted(rows.keys()):

            line = sorted(
                rows[row],
                key=lambda x: self.center_x(x["box"])
            )

            text = " ".join(i["text"] for i in line)

            lines.append(
                {
                    "text": text,
                    "items": line
                }
            )

        return lines

    def center_x(self, box):
        return (box[0][0]+box[1][0])/2

    def center_y(self, box):
        return (box[0][1]+box[2][1])/2