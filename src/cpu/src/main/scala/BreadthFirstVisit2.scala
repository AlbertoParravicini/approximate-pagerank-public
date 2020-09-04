package benchmark

import java.lang.System.currentTimeMillis

import scala.collection.JavaConverters._
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import oracle.pgx.api.internal.AnalysisResult
import oracle.pgx.common.types.PropertyType

import scala.collection.mutable


class BreadthFirstVisit2(session: PgxSession, graph: PgxGraph, maxVertices: Long) extends Algorithm  {

  val analyst = session.createAnalyst()

  val srcBFS = classOf[BreadthFirstVisit2].getResourceAsStream("/bfs2.gm")
  val bfs2 = session.compileProgram(srcBFS)  


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
    bfs2.run(graph, maxVertices.asInstanceOf[java.lang.Long], graph.getVertexProperty("dist"))
    return currentTimeMillis - start
  }
}