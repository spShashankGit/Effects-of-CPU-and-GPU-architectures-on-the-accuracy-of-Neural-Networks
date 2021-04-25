import uuid
from typing import Optional, Union

from pydantic import BaseModel

from pypads.model.models import BaseStorageModel, ResultType


class LibraryModel(BaseStorageModel):
    """
    Representation of a package or library
    """
    category: str = "Software"
    name: str = ...
    version: str = ...
    extracted: bool = False
    storage_type: Union[str, ResultType] = ResultType.library

    # @root_validator
    # def set_default(cls, values):
    #     values['_id'] = ".".join([values["name"], values["version"]])
    #     return values


class LibSelectorModel(BaseModel):
    """
    Representation of a selector for a package of library
    """
    name: str = ...  # Name of the package. Either a direct string or a regex.
    constraint: str  # Constraint for the version number
    regex: bool = False  # Flag if the name of the selector is to be considered as a regex
    specificity: int  # How specific the selector is ( important for a css like mapping of multiple selectors)

    def __hash__(self):
        return hash((self.name, self.constraint, self.specificity))

    class Config:
        orm_mode = True


class MappingModel(BaseStorageModel):
    """
    Representation of a mapping of a library
    """
    category: str = "SoftwareMapping"
    uid: Union[str, uuid.UUID] = ...
    name: str = ...
    author: Optional[str] = ...
    version: str = ...
    lib: LibSelectorModel = ...
    mapping_file: Optional[str] = ...  # reference to the mapping file artifact
    storage_type: Union[str, ResultType] = ResultType.mapping

    # @root_validator
    # def set_default(cls, values):
    #     values['_id'] = persistent_hash(
    #         (values["author"], values["version"], values["lib"].name, values["lib"].constraint))
    #     return values

    class Config:
        orm_mode = True

# class ExperimentModel(OntologyEntry):  # TODO
#     """
#     Model of the Experiment
#     """
#     name: str = ...
#     experiment = ...
#
#
# class RunModel(OntologyEntry):  # TODO
#     id: str = ...
#     name: str = ...
#     run = ...
