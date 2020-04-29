import tempfile
from plumbum import local
from zest import zest
from plaster.tools.assets import assets
from plaster.tools.schema import check


def zest_validate_job_folder():
    job_name = "__test_job1"
    job1 = local.path("./jobs_folder") / job_name
    run1 = local.path("./jobs_folder") / job_name / "run1"

    def _before():
        job1.mkdir()
        run1.mkdir()

    def _after():
        job1.delete()
        run1.mkdir()

    def it_accepts_folder_in_jobs_folder_by_absolute_str():
        job = str(local.path(local.env["ERISYON_ROOT"]) / "jobs_folder" / job_name)
        assert assets.validate_job_folder(job) == job_name

    def it_accepts_folder_in_jobs_folder_by_absolute_plumbum():
        job = local.path(local.env["ERISYON_ROOT"]) / "jobs_folder" / job_name
        assert assets.validate_job_folder(job) == job_name

    def it_accepts_folder_in_jobs_folder_by_relative_string():
        assert assets.validate_job_folder(f"./jobs_folder/{job_name}") == job_name

    def it_accepts_folder_in_jobs_folder_by_relative_string_with_trailing_slash():
        assert assets.validate_job_folder(f"./jobs_folder/{job_name}/") == job_name

    def it_accepts_named_job_folder_if_it_exists():
        assert assets.validate_job_folder(job_name) == job_name

    def it_accepts_slash_slash_jobs_folder():
        assert assets.validate_job_folder(f"//jobs_folder/{job_name}") == job_name

    def it_accepts_run_folder_if_specified():
        assert (
            assets.validate_job_folder(
                f"./jobs_folder/{job_name}/run1", allow_run_folders=True
            )
            == f"{job_name}/run1"
        )

    def it_raises_on_run_folder_if_not_specified():
        with zest.raises(ValueError):
            assets.validate_job_folder(
                f"./jobs_folder/{job_name}/run1", allow_run_folders=False
            )

    def it_raises_on_non_existing_file_in_jobs_folder():
        with zest.raises(FileNotFoundError):
            assets.validate_job_folder("./jobs_folder/__does_not_exist")

    def it_raises_on_file_outside_jobs_folder():
        with zest.raises(ValueError):
            assets.validate_job_folder("/tmp/foo/bar")

    def it_raises_on_non_str():
        with zest.raises(check.CheckError):
            assets.validate_job_folder(123)

    zest()
