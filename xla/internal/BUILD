load("@bazel_skylib//:bzl_library.bzl", "bzl_library")
load("//xla/internal:package_groups.bzl", "xla_internal_packages")

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = [":__subpackages__"],
    licenses = ["notice"],
)

# Required to load the package groups.
xla_internal_packages()

bzl_library(
    name = "package_groups_bzl",
    srcs = ["package_groups.bzl"],
    parse_tests = False,
    visibility = ["//visibility:private"],
)
