from plumbum import local
from plumbum.path.local import LocalPath
from plumbum import local, FG
from plaster.tools.schema import check


def get_user():
    user = local.env.get("RUN_USER")
    if user is None or user == "":
        raise Exception("User not found in $USER")
    return user


def jobs_folder():
    return local.path(local.env["HOST_JOBS_FOLDER"])


def jobs_folder_as_str():
    return str(jobs_folder())


def validate_job_folder(job_folder, allow_run_folders=False):
    """
    job_folder can be:
        * Canonical (relative):
            ./jobs_folder/job_folder

        * Stand-alone, in which case it is assumed to be ./jobs_folder/job_folder
            job_folder

        * URL-like, in which case it is convert to ./jobs_folder/job_folder
            //jobs_folder/job_folder

        * Absolute (must be in same as jobs_folder)
            ${ERISYON_ROOT}/jobs_folder/job_folder

        Run folders are optionally allowed.
            ./jobs_folder/job_folder/run

        DEPRECATED:
            /path/to/file (If there is already a symlink in ./jobs_folder to this file)

    Returns:
        The job_folder alone (without ./jobs_folder) or job_folder

    Raises:
        On any unrecognized form.

    """
    root_jobs_folder = jobs_folder()

    # NORMALIZE into string forms
    if isinstance(job_folder, LocalPath):
        # If plumbum style, path must be absolute
        if not str(job_folder).startswith("/"):
            raise ValueError(f"job_folder passed by plumbum path must be absolute")
        job_folder = str(job_folder)

    check.t(job_folder, str)

    # CONVERT URL-like form into canonical form
    if job_folder.startswith("//jobs_folder/"):
        job_folder = "./jobs_folder/" + job_folder[len("//jobs_folder/") :]

    # CONVERT stand-alone into canonical. Referenced directory must be in the root_jobs_folder
    if "/" not in job_folder:
        job_folder = "./jobs_folder/" + job_folder

    # CONVERT absolute to relative
    if job_folder.startswith(jobs_folder_as_str()):
        job_folder = "./jobs_folder" + job_folder[len(jobs_folder_as_str()) :]

    if not job_folder.startswith("./"):
        raise ValueError(
            f"job_folder canonical form starts with './' but found: {job_folder}"
        )

    # Now in canonical form, convert to absolute path
    abs_path = (
        root_jobs_folder / job_folder[len("./jobs_folder/") :]
    )  # Strip "./jobs_folder"

    def check_exists():
        if not abs_path.exists():
            raise FileNotFoundError("Unknown job or run folder")

    parts = job_folder.split("/")[2:]  # Skip the initial ".", "jobs_folder"
    if parts[-1] == "":
        del parts[-1]
    n_parts = len(parts)
    if allow_run_folders:
        if n_parts > 2:
            raise ValueError(
                f"{job_folder} is too many levels deep for a job_folder spec."
            )

        if n_parts == 2:
            check_exists()
            return f"{parts[0]}/{parts[1]}"

        if n_parts == 1:
            check_exists()
            return parts[0]

    else:
        if n_parts != 1:
            raise ValueError(
                f"{job_folder} is too many levels deep for a job_folder spec."
            )

        check_exists()
        return parts[0]


def validate_job_folder_return_path(job_folder, allow_run_folders=False):
    # Like validate_job_folder but returns complete plumbum path object
    return jobs_folder() / validate_job_folder(job_folder, allow_run_folders)
