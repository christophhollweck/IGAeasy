"""Run the public Nitsche comparisons; select studies below."""
from IGAeasy.examples.shells.verification.benchmarks import simply_supported_plate_convergence as plate
from IGAeasy.examples.shells.verification.benchmarks import scordelis_roof_convergence as roof
from IGAeasy.examples.shells.verification.benchmarks import pinched_hemisphere_convergence as hemisphere

studies = (plate, roof, hemisphere)


def main(show=True):
    results = {}
    for study in studies:
        print(study.__name__, flush=True)
        results[study.__name__] = study.main(show=show)
    return results


if __name__ == '__main__':
    main()
