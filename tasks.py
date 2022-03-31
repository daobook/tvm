from invoke import Collection
from d2py.tools.write import site

api = site(name='api', target='output/html')
namespace = site(name='docs', target='output/html')
namespace.add_collection(api, 'api')
