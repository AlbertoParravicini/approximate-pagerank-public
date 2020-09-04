package benchmark

import java.lang.System.currentTimeMillis

import scala.collection.JavaConverters._
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import oracle.pgx.api.internal.AnalysisResult
import oracle.pgx.common.types.PropertyType

import scala.collection.mutable


class BreadthFirstVisit(session: PgxSession, graph: PgxGraph, source: Long) extends Algorithm  {

  val analyst = session.createAnalyst()

  val srcBFS = classOf[BreadthFirstVisit].getResourceAsStream("/bfs.gm")
  val bfs = session.compileProgram(srcBFS)  


  /////////////////////////////////////
  /////////////////////////////////////

  /**
    * Compute BreadthFirstVisit using Green-Marl on an input graph.
    */
  def compute(): Long = {

    if (!graph.getVertexProperties.contains("dist")) {
      graph.createVertexProperty(PropertyType.INTEGER, "dist")
    } else{
      graph.getVertexProperty("dist").fill(0)
    }

    var start = currentTimeMillis
    bfs.run(graph, graph.getVertex(source), graph.getVertexProperty("dist"))
    return currentTimeMillis - start
  }
}
