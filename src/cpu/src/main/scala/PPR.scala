package benchmark

import java.io.{BufferedReader, File, InputStream, InputStreamReader}
import java.lang.System.currentTimeMillis
import java.nio.file.Paths

import scala.collection.JavaConverters._
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import oracle.pgx.api.internal.AnalysisResult
import oracle.pgx.common.types.PropertyType
import oracle.pgx.config.{FileGraphConfig, Format}
import org.apache.commons.io.FileUtils

import scala.collection.mutable


class PPR(
            session: PgxSession,
            graph: PgxGraph,
            graphName: String = "graph",
            minError: Double = PPR.DEFAULT_MIN_ERROR,
            maxIterationCount: Int = PPR.DEFAULT_MAX_ITERATIONS,
	    dampingFactor: Double = PPR.DEFAULT_DAMPING_FACTOR,
	    var personalizationVertex: Long = PPR.DEFAULT_PERSONALIZATION_VERTEX,
            outputDir: String) extends Algorithm {

  val analyst = session.createAnalyst()

  val src = classOf[PPR].getResourceAsStream("/ppr.gm")
  val algorithm = session.compileProgram(src)

  /////////////////////////////////////
  /////////////////////////////////////

  /**
    * Compute Personalized PageRank using Green-marl on an input graph.
    * Results can be stored in the graph file;
    */
  def compute(): Long = {

    if (!graph.getVertexProperties.contains("rank")) {
      graph.createVertexProperty(PropertyType.DOUBLE, "rank")
    } else{
      graph.getVertexProperty("rank").fill(0d)
    }

    var start = currentTimeMillis
    algorithm.run(
      graph,
      graph.getVertex(personalizationVertex),
      BigDecimal(this.dampingFactor).bigDecimal,
      this.maxIterationCount.asInstanceOf[java.lang.Integer],
      true.asInstanceOf[java.lang.Boolean],
      graph.getVertexProperty("rank")
    )
    val execTime = currentTimeMillis - start

    // Write graph to disk;
    if (!outputDir.trim.isEmpty) {
        start = currentTimeMillis
        writeGraph(graph, f"${graphName}_pr")
        println(f"Graph stored to disk, exec. time ${(currentTimeMillis - start)} ms")
    }
    return execTime
  }

  /////////////////////////////////////
  /////////////////////////////////////

  /**
    * Store the given graph to disk, as edgelist.
    *
    * @param graph  a graph to store
    * @param graphName  the name of the graph
    */
  def writeGraph(graph: PgxGraph, graphName: String): Unit = {
    try {
      val pathEdgeList = Paths.get(outputDir, f"$graphName.edgelist").toString
      val pathConfig = Paths.get(outputDir, f"$graphName.json").toString
      println(s"\t\tWriting files: $pathEdgeList; $pathConfig")

      val config = graph.store(Format.EDGE_LIST, pathEdgeList, true)
      // Fix the graph path, by default it is used the path relative to the execution directory,
      // but the path has to be the same where the configuration file is stored;
      val graphPath = config.getVertexUris.get(0)
      config.getValues.put(FileGraphConfig.Field.VERTEX_URIS, Array[String](new File(graphPath).getName))
      FileUtils.write(new File(pathConfig), config.toString, "UTF-8")
      println(f"\t\tWritten files: $pathEdgeList; $pathConfig")
    } catch {
      case e: Exception =>
        println(f"ERROR IN WRITING FILES $outputDir/$graphName")
        throw new RuntimeException(e)
    }
  }
}

object PPR {
  val DEFAULT_MIN_ERROR: Double = 10e-6
  val DEFAULT_MAX_ITERATIONS: Int = 100
  val DEFAULT_DAMPING_FACTOR: Double = 0.85d
  val DEFAULT_PERSONALIZATION_VERTEX: Long = 0
}
