Using Persistent and Interface Structures
=========================================

All persistent structures must inherit from ``BaseStruct``, which is based on
Pydantic's ``BaseModel``. ``BaseStruct`` provides the fields required to identify
a structure and class-level metadata used by the storage layer.

A typical structure is defined by extending ``BaseStruct`` and declaring its
own data fields:

.. code-block:: python

   from typing import Literal
   from aideforge.structs import BaseStruct


   class Settings(BaseStruct):
       kind: Literal["settings"] = "settings"
       version: Literal[1] = 1

       name: str
       description: str = ""

Use a distinct ``kind`` to identify each logical structure type. The ``version``
field identifies the schema version of the stored data. The examples below use
``Literal`` values so each model explicitly declares its kind and version.

Common fields and metadata
--------------------------

Every ``BaseStruct`` has the following fields and class-level metadata:

Attributes to Store
^^^^^^^^^^^^^^^^^^^

``kind`` (``str``)
   The logical category of the structure. It identifies which kind of data the
   structure represents.

``version`` (``int``)
   The schema version of the structure. Follow the versioning rules described
   below when changing the structure.

Meta information
^^^^^^^^^^^^^^^^

``scope`` (``ClassVar[Literal["global", "user"]]``, default ``"user"``)
   Controls visibility in storage. ``"global"`` structures are shared across
   users; ``"user"`` structures are isolated per user.

``storage_name`` (``ClassVar[str]``, default ``"{struct.kind}"``)
   Format string used to derive the default storage name. It is interpolated
   with the structure instance as the ``struct`` argument.

``storage_mode`` (``ClassVar[Literal["snapshot", "incremental"]]``,
default ``"snapshot"``)
   Controls how changes are stored. In ``"snapshot"`` mode, the stored object
   is overwritten. In ``"incremental"`` mode, the original object is preserved
   and changes are stored incrementally.

``kinds`` (``ClassVar[str | None]``, default ``None``)
   Optional subdirectory prefix for grouping related structures. If ``None``,
   structures are stored directly under the root search path.

These metadata values are class variables, not ordinary fields in the stored
Pydantic model. Override them on a subclass when its storage behavior requires
different settings.

Incremental structures
----------------------

Most structures use snapshot storage. Structures that need to preserve a
sequence of changes can instead inherit from ``BaseIncrementalStruct`` and use
``storage_mode = "incremental"``. Such structures have an ``events`` field
whose ``BaseEvent`` entries are stored incrementally.

One event variant is ``ChangeProperty``. It represents a change to structure's
values using an operation such as ``"replace"``, ``"append"``, ``"extend"``,
``"update"``, or ``"add"``. The exact event payload and application behavior
depend on the relevant event model.

Versioning and Backwards Compatibility
--------------------------------------

This section describes the versioning strategy for Pydantic-based structures.
The goal is to preserve compatibility with previously stored data while allowing
the structure to evolve.

Versioning rule
^^^^^^^^^^^^^^^

**Increase the version only when a change breaks compatibility.**

Adding a field with a default value does not, by itself, require a version
increase. Older data can still be loaded because Pydantic supplies the default
for the missing field.

For example, if version 1 contains ``name`` and version 2 adds an optional
``description`` field with a default value, the version can remain unchanged:

.. code-block:: python

   from pydantic import BaseModel

   class Settings(BaseModel):
       version: int = 1
       name: str
       description: str = ""

Existing data such as ``{"version": 1, "name": "demo"}`` remains valid. The
missing ``description`` field receives its default value.

When to introduce a new version
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Introduce a new version when an incompatible change means that data written
using the old structure cannot be interpreted correctly as the new structure
without an explicit migration.

Keep the old model available so that stored data using that version can still
be parsed. The version field is used as a discriminator to select the
appropriate model.

Example: migrating version 1 to version 2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Suppose version 1 stored a single ``owner`` field, while version 2 stores the
same information in a ``created_by`` field. This is a breaking change, so
version 2 is introduced and a migration converts version 1 data to the current
model.

.. code-block:: python

   from typing import Annotated, Literal

   from pydantic import BaseModel, Field, RootModel


   class SettingsV1(BaseModel):
       version: Literal[1] = 1
       name: str
       owner: str


   class Settings(BaseModel):
       version: Literal[2] = 2
       name: str
       created_by: str
       description: str = ""


   AnySettings = Annotated[
       SettingsV1 | Settings,
       Field(discriminator="version"),
   ]


   class SettingsVersionManager(RootModel[Settings]):
       @classmethod
       def parse_and_migrate(cls, raw_data: dict) -> Settings:
           adapter = RootModel[AnySettings]
           parsed_settings = adapter.model_validate(raw_data).root

           if isinstance(parsed_settings, SettingsV1):
               return cls._migrate_v1_to_v2(parsed_settings)

           return parsed_settings

       @staticmethod
       def _migrate_v1_to_v2(v1: SettingsV1) -> Settings:
           return Settings(
               version=2,
               name=v1.name,
               created_by=v1.owner,
               description="",
           )


The example uses ``Literal`` values for the version fields. This makes each
model's version explicit and allows Pydantic to select a model using the
``version`` discriminator.

How to use the version manager
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Pass the raw dictionary to ``parse_and_migrate`` rather than validating it
directly as the current ``Settings`` model. The manager first parses the data
using the model that matches its version, then migrates older data to the
current model.

.. code-block:: python

   old_data = {
       "version": 1,
       "name": "demo",
       "owner": "Ferenc",
   }

   settings = SettingsVersionManager.parse_and_migrate(old_data)

   assert isinstance(settings, Settings)
   assert settings.version == 2
   assert settings.created_by == "Ferenc"

The returned value is always the current ``Settings`` model. Callers therefore
do not need to handle each historical version themselves.

Adding another version
^^^^^^^^^^^^^^^^^^^^^^

When a later breaking change is needed:

#. Define a new model for the new version.
#. Add that model to the discriminated union.
#. Add a migration function from the previous version to the new current
   version.
#. Update the version manager so that every supported older version is
   migrated to the current model.

Keep migrations explicit and test them with representative data from every
supported historical version.

Key points
^^^^^^^^^^

* Do not increase the version for a compatible change, such as adding a field
  with a suitable default value.
* Increase the version for a breaking change that requires migration.
* Keep historical models available for parsing previously stored data.
* Use the version field as a discriminator to identify the stored format.
* Migrate older data at the boundary where it is loaded, so the rest of the
  application works with the current model only.
