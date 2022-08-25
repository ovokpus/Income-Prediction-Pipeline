from prefect.dployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner

DeploymentSpec(
    flow_location="score.py",
    name="income_prediction",
    parameters={
        "run_id": "run_id"
    },
    flow_storage="",
    schedule=CronSchedule(cron="0 0 * * *"),
    flow_runner=SubprocessFlowRunner(),
    tags=["ml", "xgboost"]
)
