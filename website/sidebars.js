module.exports = {
  usersSidebar: {
    "Getting Started": ['users/intro', 'users/installation', 'users/start'],
    "Normalizing Flows": ['users/univariate', 'users/multivariate', 'users/conditional'],
    "Basic Concepts": ['users/shapes', 'users/constraints', 'users/bijectors', 'users/parameters', 'users/transformed_distributions', 'users/composing'],
    "Advanced Topics": ['users/caching', 'users/initialization', 'users/structure', 'users/torchscript'],
  },
  devsSidebar: {
    "General": ['dev/why', 'dev/contributing', 'dev/code_of_conduct', 'dev/about'],
    "Extending the Library": ['dev/overview', 'dev/docs', 'dev/tests', 'dev/bijector', 'dev/param'],
    "Literature Survey": ['dev/prior', 'dev/libraries', 'dev/methodology', 'dev/applications'],
  },
  apiSidebar: {
    "Python API": ['api/flowtorch', 'api/bijectors', 'api/distributions', 'api/params', 'api/utils'],
  },
};
