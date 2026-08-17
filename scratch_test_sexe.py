from app.pipeline import DocumentPipeline
import pprint

pipeline = DocumentPipeline(fast_mode=True)
res = pipeline.process("images/female_1.jpg")
pprint.pprint(res["data"])
