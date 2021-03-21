from kountServer.server import app
import logging
if __name__ == "__main__":
  app.run()
  app.logger.addHandler(logging.StreamHandler(sys.stdout))
  app.logger.setLevel(logging.ERROR)
