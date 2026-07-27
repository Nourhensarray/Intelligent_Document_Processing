from app.quality.image_quality_checker import ImageQualityChecker


images = [
    "images/test.jpeg",
    "images/cin2.jpg",
    "images/cin3.jpg",
]


checker = ImageQualityChecker()


for image_path in images:

    result = checker.check(
        image_path
    )

    print()
    print("=" * 60)
    print(f"IMAGE : {image_path}")
    print("=" * 60)

    print(
        f"STATUS : {result['status']}"
    )

    print(
        f"SCORE : {result['score']}/100"
    )

    print()

    print("METRICS :")

    for name, metric in result["metrics"].items():

        print(
            f"{name} : {metric}"
        )

    print()

    print(
        "REASONS :",
        result["reasons"]
    )