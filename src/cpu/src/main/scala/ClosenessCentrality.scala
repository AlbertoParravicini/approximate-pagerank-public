package benchmark

import java.lang.System.currentTimeMillis

import scala.collection.JavaConverters._
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import oracle.pgx.api.internal.AnalysisResult
import oracle.pgx.common.types.PropertyType
import java.util.Map

import scala.collection.mutable


class ClosenessCentrality(session: PgxSession, graph: PgxGraph, maxVertices: Long) extends Algorithm  {

  val analyst = session.createAnalyst()

  val src = classOf[ClosenessCentrality].getResourceAsStream("/cc.gm")
  val algorithm = session.compileProgram(src)  


  /////////////////////////////////////
  /////////////////////////////////////

  /**
    * Compute ClosenessCentrality using Green-Marl on an input graph.
    */
  def compute(): Long = {

    if (!graph.getVertexProperties.contains("cc")) {
      graph.createVertexProperty(PropertyType.DOUBLE, "cc")
    } else{
      graph.getVertexProperty("cc").fill(0d)
    }

    var start = currentTimeMillis
    algorithm.run(graph, maxVertices.asInstanceOf[java.lang.Long], graph.getVertexProperty("cc"))
    
    return currentTimeMillis - start
  }
}