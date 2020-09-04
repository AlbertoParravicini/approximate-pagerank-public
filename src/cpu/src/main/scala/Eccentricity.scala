package benchmark

import java.lang.System.currentTimeMillis

import scala.collection.JavaConverters._
import oracle.pgx.api.filter.{EdgeFilter, VertexFilter}
import oracle.pgx.api._
import oracle.pgx.api.internal.AnalysisResult
import oracle.pgx.common.types.PropertyType

import scala.collection.mutable


class Eccentricity(session: PgxSession, graph: PgxGraph, maxVertices: Long) extends Algorithm  {

  val analyst = session.createAnalyst()

  val src = classOf[Eccentricity].getResourceAsStream("/eccentricity.gm")
  val algorithm = session.compileProgram(src)  


  /////////////////////////////////////
  /////////////////////////////////////

  /**
    * Compute Eccentricity using Green-Marl on an input graph.
    */
  def compute(): Long = {

    if (!graph.getVertexProperties.contains("ecc")) {
      graph.createVertexProperty(PropertyType.INTEGER, "ecc")
    } else{
      graph.getVertexProperty("ecc").fill(0)
    }

    var start = currentTimeMillis
    algorithm.run(graph, maxVertices.asInstanceOf[java.lang.Long], graph.getVertexProperty("ecc"))
    return currentTimeMillis - start
  }
}