package benchmark

import java.lang.System.currentTimeMillis
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import java.util.concurrent.ThreadLocalRandom

object Main {

  val DEFAULT_GRAPH_PATH: String = "../../data/graphs/edgelist/graph_small_c16.json"
  val DEFAULT_OUTPUT_DIR: String = ""
  val DEFAULT_UNDIRECT = false
  val DEFAULT_SOURCE: Long = -1
  val DEFAULT_NUM_TESTS = 10
  val DEFAULT_MAX_VERTICES: Long = -1
  val DEFAULT_DEBUG: Boolean = false

  def main(args: Array[String]): Unit = {

    var algorithm: Option[String] = None
    // Path to the input file;
    var inputPath = DEFAULT_GRAPH_PATH
    var minError = PageRank.DEFAULT_MIN_ERROR
    var maxIterationCount = PageRank.DEFAULT_MAX_ITERATIONS
    var outputDir = DEFAULT_OUTPUT_DIR
    var undirect = DEFAULT_UNDIRECT
    var source = DEFAULT_SOURCE
    var numTests = DEFAULT_NUM_TESTS
    var maxVertices: Long = DEFAULT_MAX_VERTICES
    var dampingFactor: Double = PageRank.DEFAULT_DAMPING_FACTOR
    var debug: Boolean = DEFAULT_DEBUG

    var pprSources: Seq[Long] = Seq(0, 0, 0, 0, 0, 0, 0, 0)

    var useRandomSource: Boolean = false

    if (debug) {
      println("Starting Pgx CPU Benchmark... ")
    }

    parser.parse(args, Config()) match {
      case Some(config) =>
        inputPath = config.inputPath
        undirect = config.undirect
        minError = config.minError
        maxIterationCount = config.maxIterationCount
        outputDir = config.outputDir
        algorithm = config.algorithm
        source = config.source
        numTests = config.numTests
        maxVertices = config.maxVertices
        dampingFactor = config.dampingFactor
        debug = config.debug
      case None =>
        println(f"Invalid input parameters: ${args}")
        println("ENDING BENCHMARK")
        return
    }
    if (debug) {
      println(f"\tInput graph: ${inputPath}")
    }
    val name = "graph"

    val algorithmName = algorithm.getOrElse("")
    if (debug) {
      println(f"Chosen algorithm: $algorithmName")
    }

    if (debug) {
      println("Creating Pgx session...")
    }
    val session: PgxSession = Pgx.createSession("session_1")
    if (debug) {
      println("Session created")
    }

    // Load the graph;
    var startTime = currentTimeMillis
    if (debug) {
      println(f"Opening input graph: ${inputPath}...")
    }
    var graph = session.readGraphWithProperties(inputPath);
    if (debug) {
      println("Loaded input graph")
      println(f"\tGraph loading time: ${(currentTimeMillis - startTime) / 1000} sec")
    }
    // Undirect the graph;
    if (undirect) {
      if (debug) {
        println("Undirecting graph...")
      }
      startTime = currentTimeMillis
      graph = graph.undirect()
      if (debug) {
        println(f"\tGraph undirection time: ${(currentTimeMillis - startTime) / 1000} sec")
      }
    }

    // Check if the source is correct, or get a random one;
    if (source < 0 || source >= graph.getNumVertices) {
      useRandomSource = true
      val oldSource = source
      source = ThreadLocalRandom.current().nextLong(graph.getNumVertices)
      if (debug) {
        println(s"invalid source: $oldSource, using new random source $source")
      }
    }
    if (algorithmName == "ppr2") {
      pprSources = (1 to 8).map { _ => ThreadLocalRandom.current().nextLong(graph.getNumVertices) }
      if (debug) {
        println(s"using ppr sources: $pprSources")
      }
    }

    if (debug) {
      println(f"- V: ${graph.getNumVertices}")
      println(f"- E: ${graph.getNumEdges}")
    }
    // Define the max. number of vertices on which the computation is performed, on some algorithms;
    if (maxVertices > 0) {
      maxVertices = Math.min(maxVertices, graph.getNumVertices)
      if (debug) {
        println(f"Computing algorithm on $maxVertices vertices")
      }
    } else {
      maxVertices = graph.getNumVertices
    }

    // Run the benchmark;
    var alg: Option[Algorithm] = None
    if (algorithmName == "pagerank") {
      val pr = new PageRank(
        session = session,
        graph = graph,
        graphName = name,
        minError = minError,
        maxIterationCount = maxIterationCount,
        outputDir = outputDir,
        dampingFactor = dampingFactor
      )
      alg = Some(pr)
    } else if (algorithmName == "ppr") {
      val ppr = new PPR(
        session = session,
        graph = graph,
        graphName = name,
        minError = minError,
        maxIterationCount = maxIterationCount,
        outputDir = outputDir,
        dampingFactor = dampingFactor,
        personalizationVertex = source
      )
      alg = Some(ppr)
    } else if (algorithmName == "ppr2") {
      val ppr = new PPR2(
        session = session,
        graph = graph,
        graphName = name,
        minError = minError,
        maxIterationCount = maxIterationCount,
        outputDir = outputDir,
        dampingFactor = dampingFactor,
        personalizationVertices = pprSources
      )
      alg = Some(ppr)
    } else if (algorithmName == "bfs") {
      if (debug) {
        println(f"Running BFS - Source: $source")
      }
      val bfs = new BreadthFirstVisit(session, graph, source)
      alg = Some(bfs)

    } else if (algorithmName == "bfs2") {
      if (debug) {
        println(f"Running BFS from $maxVertices sources")
      }
      val bfs2 = new BreadthFirstVisit2(session, graph, maxVertices)
      alg = Some(bfs2)

    } else if (algorithmName == "fbfs") {
      if (debug) {
        println(f"Running FBFS - Source: $source")
      }
      val fbfs = new FilteredBFS(session, graph, source)
      alg = Some(fbfs)

    } else if (algorithmName == "cc") {
      if (debug) {
        println(f"Running Closesess Centrality - Max vertices: $maxVertices")
      }
      val cc = new ClosenessCentrality(session, graph, maxVertices)
      alg = Some(cc)

    } else if (algorithmName == "eccentricity") {
      if (debug) {
        println(f"Running Eccentricity - Max vertices: $maxVertices")
      }
      val eccentricity = new Eccentricity(session, graph, maxVertices)
      alg = Some(eccentricity)

    } else {
      println(f"ERROR, invalid algorithm selected: $algorithmName!")
      return
    }

    if (alg.isDefined) {
      for (i <- 0 to numTests) {

        // Update random source;
        if (algorithmName == "ppr") {
          val oldSource = source
          source = ThreadLocalRandom.current().nextLong(graph.getNumVertices)
          alg.get.asInstanceOf[PPR].personalizationVertex = source
          if (debug) {
            println(s"invalid source: $oldSource, using new random source ${source}")
          }
        } else if (algorithmName == "ppr2") {
          pprSources = (1 to 8).map { _ => ThreadLocalRandom.current().nextLong(graph.getNumVertices) }.toSeq
          alg.get.asInstanceOf[PPR2].personalizationVertices = pprSources
          if (debug) {
            println(s"using ppr sources: $pprSources")
          }
        }
        
        val execTime = alg.get.compute()
        if (debug) {
          println(f"$algorithmName, $execTime ms")
        } else if (algorithmName == "pagerank") {
          println(s"0,0,$inputPath,${graph.getNumVertices},${graph.getNumEdges},$execTime,0,0,0")
        } else if (algorithmName == "ppr") {
          println(s"0,0,$inputPath,${graph.getNumVertices},${graph.getNumEdges},$source,$execTime,0,0,0")
        } else if (algorithmName == "ppr2") {
          println(s"0,0,$inputPath,${graph.getNumVertices},${graph.getNumEdges},${pprSources.mkString(";")},$execTime,0,0,0")
        }
      }
    }
  }

  /////////////////////////////////////
  /////////////////////////////////////

  // Configuration of the input parameters;
  case class Config(inputPath: String = DEFAULT_GRAPH_PATH,
                    minError: Double = PageRank.DEFAULT_MIN_ERROR,
                    maxIterationCount: Int = PageRank.DEFAULT_MAX_ITERATIONS,
                    outputDir: String = DEFAULT_OUTPUT_DIR,
                    undirect: Boolean = DEFAULT_UNDIRECT,
                    debug: Boolean = DEFAULT_DEBUG,
                    algorithm: Option[String] = None,
                    source: Long = DEFAULT_SOURCE,
                    numTests: Int = DEFAULT_NUM_TESTS,
                    maxVertices: Long = DEFAULT_MAX_VERTICES,
                    dampingFactor: Double = PageRank.DEFAULT_DAMPING_FACTOR
                   )

  // Define a command line args parse to handle the input file
  // and other parameters;
  val parser: scopt.OptionParser[Config] = new scopt.OptionParser[Config]("benchmark-main") {
    head("benchmark-main", "0.1")

    opt[String]('g', "graph").valueName("path/to/graph")
      .action((x, c) => c.copy(inputPath = x)).text("path to the graph configuration file")
      .withFallback(() => DEFAULT_GRAPH_PATH)

    opt[String]('n', "algorithm").valueName("name-of-algorithm")
      .action((x, c) => c.copy(algorithm = Some(x))).text("name of the algorithm to benchmark")

    opt[Double]('e', "min-error")
      .action((x, c) => c.copy(minError = x)).text("tolerance value used to establish convergence")
      .withFallback(() => PageRank.DEFAULT_MIN_ERROR)
      .validate(x =>
        if (x >= 0f) success
        else failure("value must be >= 0"))

    opt[Double]('a', "alpha")
      .action((x, c) => c.copy(dampingFactor = x)).text("damping factor used in pagerank")
      .withFallback(() => PageRank.DEFAULT_DAMPING_FACTOR)
      .validate(x =>
        if (x >= 0f) success
        else failure("value must be >= 0"))

    opt[Int]('m', "max-iterations")
      .action((x, c) => c.copy(maxIterationCount = x)).text("maximum number of iterations ot be performed")
      .withFallback(() => PageRank.DEFAULT_MAX_ITERATIONS)
      .validate(x =>
        if (x > 0) success
        else failure("value must be > 0"))

    opt[Int]('t', "num-tests")
      .action((x, c) => c.copy(numTests = x)).text("number of times the algorithm is executed")
      .withFallback(() => DEFAULT_NUM_TESTS)
      .validate(x =>
        if (x >= 0) success
        else failure("value must be >= 0"))

    opt[Long]('v', "max-vertices")
      .action((x, c) => c.copy(maxVertices = x)).text("number of vertices on which the algorithm is executed, if possible")
      .withFallback(() => DEFAULT_MAX_VERTICES)

    opt[Unit]('u', "undirect")
      .action((_, c) => c.copy(undirect = true)).text("if present, undirect the graph")

    opt[Unit]('d', "debug")
      .action((_, c) => c.copy(debug = true)).text("if present, use debug prints")

    opt[String]('o', "output-dir").valueName("path/to/output/dir")
      .action((x, c) => c.copy(outputDir = x)).text("path to the directory where the output is stored")
      .withFallback(() => DEFAULT_OUTPUT_DIR)

    opt[Long]('s', "source-vertex").valueName("id-of-vertex")
      .action((x, c) => c.copy(source = x)).text("source vertex of BFS and Personalized PageRank")
  }
}
